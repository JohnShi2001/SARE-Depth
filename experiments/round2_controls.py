"""New Round-2 controls. Historical source/checkpoints/manifests are read only."""
import argparse,csv,datetime,hashlib,json,math,os,random,sys,time,traceback
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('SARE_EXPERIMENT_OUTPUT', str(ROOT/'outputs/round2'))).resolve()
sys.path.insert(0,str(ROOT/'vendor'))
from depth_anything_v2.dpt import DepthAnythingV2
BASE=ROOT/'weights/stage1_base.pth'
PUBLIC=Path(os.environ.get('SARE_PUBLIC_WEIGHT', str(ROOT/'weights/depth_anything_v2_metric_vkitti_vits.pth')))
DATA=Path(os.environ.get('SARE_DATA_ROOT', str(ROOT/'example_data'))).resolve()
EXPECTED={'base':'11711dc7f4ef52b24b4c52a01c4d168b76f41a7a22b463579ede31788531e8e1','public':'9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb'}
MEAN=torch.tensor([.485,.456,.406])[:,None,None];STD=torch.tensor([.229,.224,.225])[:,None,None]

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def table(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 torch.set_num_threads(4)
def manifest(split):return ROOT/'manifests'/f'citrusfarm_{split}.csv'
def rows(split):
 with manifest(split).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def resolve(s):
 p=Path(s)
 if p.is_absolute():return p
 if (DATA/p).is_file():return DATA/p
 return ROOT/p
class Images(Dataset):
 def __init__(self,split):self.rows=rows(split)
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  rgb=Image.open(resolve(r['image_path'])).convert('RGB').resize((518,518),Image.Resampling.BILINEAR)
  x=(torch.from_numpy(np.asarray(rgb).astype(np.float32)/255).permute(2,0,1)-MEAN)/STD
  g=torch.from_numpy(np.load(resolve(r['depth_path'])).astype(np.float32))
  if g.ndim==3:g=g[...,0]
  g=F.interpolate(g[None,None],(518,518),mode='nearest')[0,0]
  return x,g,r['sample_id']
def load(kind):
 path=BASE if kind=='base' else PUBLIC
 assert sha(path)==EXPECTED[kind]
 payload=torch.load(path,map_location='cpu',weights_only=False)
 state=payload.get('model',payload.get('state_dict',payload))
 config=payload.get('config',{})
 if kind=='base':assert float(config.get('max_depth',80))==80
 m=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384],max_depth=80,
   enable_focal_head=kind=='base',focal_head_type='df_b' if kind=='base' else 'none')
 info=m.load_state_dict(state,strict=True)
 return m.cuda().eval(),{'checkpoint':str(path),'sha256':sha(path),'missing':list(info.missing_keys),'unexpected':list(info.unexpected_keys),'max_depth':80,'params_total_with_aux':sum(p.numel() for p in m.parameters()),'params_depth_only':sum(p.numel() for n,p in m.named_parameters() if not n.startswith('focal_head.'))}
def depth(m,x):
 y=m(x);return y['depth'] if isinstance(y,dict) else y
def tensor_hash(items):
 h=hashlib.sha256()
 for n,t in sorted(items):h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def freeze(m):
 for n,p in m.named_parameters():p.requires_grad_(n.startswith('depth_head.'))
 m.eval();m.depth_head.train()
 return [n for n,p in m.named_parameters() if p.requires_grad]
def frozen_hash(m):return tensor_hash([(n,t) for n,t in m.state_dict().items() if not n.startswith('depth_head.')])
def head_forward(m,x):
 with torch.no_grad():feats=m.pretrained.get_intermediate_layers(x,m.intermediate_layer_idx[m.encoder],return_class_token=True)
 d=m.depth_head(feats,x.shape[-2]//14,x.shape[-1]//14)
 if isinstance(d,tuple):d=d[0]
 return (d*80).squeeze(1)
def loss(p,g):
 v=torch.isfinite(g)&(g>=.3)&(g<=30)
 assert bool(v.any()) and bool(torch.isfinite(p).all()) and bool((p>0).all())
 return (p[v].clamp_min(1e-6).log()-g[v].clamp_min(1e-6).log()).abs().mean()+.1*(p[v]-g[v]).abs().mean()
def metrics(p,g):
 v=torch.isfinite(g)&(g>=.3)&(g<=30);a=p[v].double();b=g[v].double();n=len(b)
 bad=int((~torch.isfinite(a)|(a<=0)).sum())
 if bad:raise RuntimeError(f'Invalid predictions on fixed GT support: {bad}/{n}')
 if not n:raise RuntimeError('No valid reference pixels')
 c=a.clamp(1e-6,30);diff=c-b;rawdiff=a-b
 return {'pixels':n,'sum_absrel':float((diff.abs()/b).sum()),'sum_sq':float(diff.square().sum()),'sum_mae':float(diff.abs().sum()),'delta1_count':int((torch.maximum(c/b,b/c)<1.25).sum()),'uncapped_sum_absrel':float((rawdiff.abs()/b).sum()),'uncapped_sum_sq':float(rawdiff.square().sum()),'uncapped_sum_mae':float(rawdiff.abs().sum()),'uncapped_delta1_count':int((torch.maximum(a/b,b/a)<1.25).sum()),'cap_hits':int((a>30).sum()),'invalid_predictions':bad,'median_raw_pred_gt_ratio':float(torch.median(a/b)),'median_raw_log_ratio':float(torch.median((a/b).log())),'raw_prediction_max':float(a.max())}
def aggregate(rec):
 n=sum(r['pixels'] for r in rec)
 return {'images':len(rec),'valid_pixels':n,'AbsRel':sum(r['sum_absrel'] for r in rec)/n,'RMSE':math.sqrt(sum(r['sum_sq'] for r in rec)/n),'MAE':sum(r['sum_mae'] for r in rec)/n,'delta1':sum(r['delta1_count'] for r in rec)/n,'uncapped_AbsRel':sum(r['uncapped_sum_absrel'] for r in rec)/n,'uncapped_RMSE':math.sqrt(sum(r['uncapped_sum_sq'] for r in rec)/n),'uncapped_MAE':sum(r['uncapped_sum_mae'] for r in rec)/n,'uncapped_delta1':sum(r['uncapped_delta1_count'] for r in rec)/n,'cap_hit_fraction':sum(r['cap_hits'] for r in rec)/n,'invalid_predictions':sum(r['invalid_predictions'] for r in rec),'mean_image_median_raw_pred_gt_ratio':float(np.mean([r['median_raw_pred_gt_ratio'] for r in rec])),'mean_image_median_raw_log_ratio':float(np.mean([r['median_raw_log_ratio'] for r in rec])),'raw_prediction_max':max(r['raw_prediction_max'] for r in rec)}
@torch.no_grad()
def evaluate(m,split,mode='square',limit=None,label='',keep=True):
 ds=Images(split);rec=[];start=time.perf_counter();m.eval()
 for i in range(len(ds) if limit is None else min(limit,len(ds))):
  x,g,sid=ds[i]
  if mode=='aspect_preserving':
   raw=cv2.imread(str(resolve(ds.rows[i]['image_path'])));assert raw is not None
   x,_=m.image2tensor(raw,518)
  else:x=x[None].cuda()
  pred=depth(m,x.float())
  pred=F.interpolate(pred[:,None],(518,518),mode='bilinear',align_corners=True)[0,0].cpu()
  rec.append({'sample_id':sid,'input_height':x.shape[-2],'input_width':x.shape[-1],**metrics(pred,g)})
  if i%50==0:put(OUT/'progress.json',{'stage':label,'processed':i+1,'total':len(ds) if limit is None else limit,'time':now()})
 res={**aggregate(rec),'seconds':time.perf_counter()-start,'mode':mode,'split':split,'input_shapes':sorted(set((r['input_height'],r['input_width']) for r in rec))}
 if keep:table(OUT/'results'/f'{label}_per_image.csv',rec);put(OUT/'results'/f'{label}_summary.json',res)
 return res
def protocol():
 return {'created_utc':now(),'runner_sha256':sha(__file__),'weights':EXPECTED,'manifests':{s:{'path':str(manifest(s)),'sha256':sha(manifest(s)),'rows':len(rows(s))} for s in ['train','val','test']},'precision':'FP32; TF32 disabled for matmul and cuDNN','grid':'518x518 reference, nearest resampling of original depth, identical GT-only finite support inclusive [0.3,30] m','square':'PIL RGB bilinear 518x518; ImageNet normalization','aspect_preserving':'original BGR image through historical DAV2 image2tensor, cubic lower_bound, multiple14; normalized RGB; bilinear output resize align_corners=True to common518 grid','evaluation':'pixel-pooled AbsRel/RMSE/MAE/delta1 with common prediction clamp[1e-6,30]; uncapped metrics also retained','scale':'arithmetic mean over images of each valid-GT-support median raw pred/GT and median raw log-ratio, before cap','invalid':'fail run on any nonfinite/nonpositive prediction on fixed GT support; do not discard prediction-dependent pixels','head_only':{'module':'full original depth_head (DPTHead), encoder and focal auxiliary frozen/eval','epochs':30,'batch_size':4,'updates_per_epoch':875,'updates_per_run':26250,'seeds':[20260711,20260712,20260713],'optimizer':'AdamW','lr_candidates':[1e-4,1e-5],'lr_selection':'one epoch per candidate, seed20260711, lowest full-validation uncapped pooledAbsRel; fresh initial checkpoint for every pilot and formal run','weight_decay':1e-4,'scheduler':None,'loss':'mean absolute log-depth + 0.1 * metric MAE; no adapter-specific regularizers','selection':'minimum uncapped validation pooledAbsRel; first epoch wins exact ties','augmentations':'none, matching historical DP-RHR dataset','historical_DP_RHR':'30epochs,batch4,AdamW1e-4,wd1e-4,AbsRel validation selection; 3500train/750val; full precision new head forward versus historical runtime as documented','timing':'new wall-time reported separately, not matched inference backend to TensorRT'}}
def smoke():
 assert not (OUT/'SMOKE.json').exists()
 put(OUT/'PROTOCOL.json',protocol())
 results={}
 for kind in ['public','base']:
  m,info=load(kind)
  results[kind]={'load':info}
  for mode in ['square','aspect_preserving']:results[kind][mode]=evaluate(m,'val',mode,2,'smoke_'+kind+'_'+mode,False)
  if kind=='base':
   whitelist=freeze(m);before=frozen_hash(m);optimizer=torch.optim.AdamW(m.depth_head.parameters(),lr=1e-4,weight_decay=1e-4)
   dl=DataLoader(Images('train'),batch_size=4,shuffle=True,num_workers=2,pin_memory=True)
   times=[];ls=[]
   for j,(x,g,_) in enumerate(dl):
    if j==24:break
    torch.cuda.synchronize();t=time.perf_counter();optimizer.zero_grad(set_to_none=True);p=head_forward(m,x.cuda());l=loss(p,g.cuda());l.backward()
    bad=[n for n,p in m.named_parameters() if not n.startswith('depth_head.') and p.grad is not None];assert not bad
    assert all(torch.isfinite(p.grad).all() for p in m.depth_head.parameters() if p.grad is not None)
    optimizer.step();torch.cuda.synchronize();times.append(time.perf_counter()-t);ls.append(float(l))
   assert before==frozen_hash(m)
   results['head_smoke']={'steps':len(times),'loss_first':ls[0],'loss_last':ls[-1],'mean_step_seconds_post_warmup':float(np.mean(times[4:])),'three_run_train_hours_estimate':float(np.mean(times[4:])*26250*3/3600),'frozen_sha256_before_after':before,'gradient_whitelist':whitelist,'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
  del m;torch.cuda.empty_cache()
 put(OUT/'SMOKE.json',results)
 print(json.dumps(results),flush=True)
def preprocessing():
 assert (OUT/'SMOKE.json').is_file()
 dest=OUT/'results/paired_preprocessing_complete.json';assert not dest.exists()
 res={}
 for kind in ['public','base']:
  m,info=load(kind)
  for mode in ['square','aspect_preserving']:
   label=kind+'_'+mode
   res[label]=evaluate(m,'test',mode,label=label)
   print(label,json.dumps(res[label]),flush=True)
  del m;torch.cuda.empty_cache()
 assert len({r['valid_pixels'] for r in res.values()})==1
 put(dest,{'status':'COMPLETE','protocol_sha256':sha(OUT/'PROTOCOL.json'),'groups':res,'completed':now()})
def train_one(s,lr,epochs,label):
 run=OUT/'head_only'/label;run.mkdir(parents=True,exist_ok=False)
 seed(s);m,info=load('base');whitelist=freeze(m);fhash=frozen_hash(m)
 put(run/'CONFIG.json',{'seed':s,'learning_rate':lr,'epochs':epochs,'load':info,'gradient_whitelist':whitelist,'frozen_sha256_before':fhash,'protocol_sha256':sha(OUT/'PROTOCOL.json')})
 opt=torch.optim.AdamW(m.depth_head.parameters(),lr=lr,weight_decay=1e-4)
 dl=DataLoader(Images('train'),batch_size=4,shuffle=True,num_workers=2,pin_memory=True)
 best=float('inf');logs=[];start=time.perf_counter();steps=0
 for ep in range(1,epochs+1):
  m.eval();m.depth_head.train();loss_sum=0.
  for j,(x,g,_) in enumerate(dl):
   opt.zero_grad(set_to_none=True);p=head_forward(m,x.cuda(non_blocking=True));l=loss(p,g.cuda(non_blocking=True));assert torch.isfinite(l)
   l.backward();assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('depth_head.'))
   assert all(torch.isfinite(p.grad).all() for p in m.depth_head.parameters() if p.grad is not None)
   opt.step();loss_sum+=float(l);steps+=1
   if j%25==0:put(OUT/'progress.json',{'stage':label,'epoch':ep,'epochs':epochs,'batch':j+1,'batches':len(dl),'steps':steps,'loss':float(l),'elapsed_sec':time.perf_counter()-start,'time':now()})
  val=evaluate(m,'val',label=label+'_val_epoch'+str(ep),keep=False)
  row={'epoch':ep,'steps':steps,'mean_loss':loss_sum/len(dl),**val};logs.append(row)
  if val['uncapped_AbsRel']<best:
   best=val['uncapped_AbsRel'];torch.save({'depth_head':m.depth_head.state_dict(),'seed':s,'epoch':ep,'validation':val,'initial_checkpoint_sha256':EXPECTED['base']},run/'best_head.pth')
  assert frozen_hash(m)==fhash
  table(run/'epochs.csv',logs);put(run/'progress.json',{'epoch':ep,'epochs':epochs,'best_val_AbsRel':best,'wall_seconds':time.perf_counter()-start,'time':now()})
  print(label,'epoch',ep,'val',val['uncapped_AbsRel'],flush=True)
 put(run/'COMPLETE.json',{'status':'COMPLETE','seed':s,'epochs':epochs,'updates':steps,'best_uncapped_val_AbsRel':best,'wall_seconds':time.perf_counter()-start,'frozen_sha256_after':frozen_hash(m),'frozen_unchanged':frozen_hash(m)==fhash,'head_checkpoint_sha256':sha(run/'best_head.pth'),'parameters_updated':sum(p.numel() for p in m.parameters() if p.requires_grad)})
 del m;torch.cuda.empty_cache();return best
def training():
 assert (OUT/'SMOKE.json').is_file()
 pilots=[]
 for lr in [1e-4,1e-5]:pilots.append({'lr':lr,'val':train_one(20260711,lr,1,'pilot_'+str(lr))})
 lr=min(pilots,key=lambda x:x['val'])['lr'];put(OUT/'head_only/LR_LOCK.json',{'pilots':pilots,'selected_lr':lr,'test_used':False,'locked':now()})
 for s in [20260711,20260712,20260713]:train_one(s,lr,30,'seed_'+str(s))
 put(OUT/'head_only/TRAINING_COMPLETE.json',{'status':'COMPLETE','seeds':[20260711,20260712,20260713],'test_evaluation_pending':True,'time':now()})
def head_test():
 assert (OUT/'head_only/TRAINING_COMPLETE.json').is_file()
 destinations=[OUT/'results/head_test_complete.json']
 for s in [20260711,20260712,20260713]:
  destinations.extend([OUT/'results'/f'head_{s}_per_image.csv',OUT/'results'/f'head_{s}_summary.json'])
 existing=[str(p) for p in destinations if p.exists()]
 if existing:raise FileExistsError('Test outputs already exist: '+', '.join(existing))
 res=[]
 for s in [20260711,20260712,20260713]:
  m,info=load('base');path=OUT/'head_only'/('seed_'+str(s))/'best_head.pth';p=torch.load(path,map_location='cpu',weights_only=False)
  m.depth_head.load_state_dict(p['depth_head'],strict=True)
  res.append({'seed':s,'selected_epoch':p['epoch'],'checkpoint_sha256':sha(path),**evaluate(m,'test',label='head_'+str(s))})
  del m;torch.cuda.empty_cache()
 put(OUT/'results/head_test_complete.json',{'runs':res,'mean_population_sd':{k:{'mean':float(np.mean([r[k] for r in res])),'sd_population':float(np.std([r[k] for r in res],ddof=0))} for k in ['AbsRel','RMSE','MAE','delta1','uncapped_AbsRel','uncapped_RMSE','uncapped_MAE','uncapped_delta1']},'time':now()})
def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'results').mkdir(exist_ok=True);seed(20260711)
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['smoke','preprocessing','training','head_test']);a=parser.parse_args()
 try:
  globals()[a.stage]();put(OUT/(a.stage+'_EXIT.json'),{'exit':0,'time':now()})
 except Exception as e:
  put(OUT/(a.stage+'_FAILURE_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'),{'error':str(e),'traceback':traceback.format_exc(),'time':now()});raise
if __name__=='__main__':main()
