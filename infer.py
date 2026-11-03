import argparse
from pathlib import Path
import numpy as np
import torch
from sare_depth.model import SAREDepth
from sare_depth.data import image_tensor

def main():
    p=argparse.ArgumentParser()
    p.add_argument("image");p.add_argument("--output",required=True)
    p.add_argument("--weights",default="weights");p.add_argument("--device",default="cpu")
    p.add_argument("--seed",type=int,default=20260711,choices=[20260711,20260712,20260713])
    a=p.parse_args();target=Path(a.output)
    if target.exists():raise FileExistsError(target)
    model=SAREDepth(a.weights,a.seed).to(a.device).eval()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with torch.inference_mode():outputs=model(image_tensor(a.image)[None].to(a.device))
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("wb") as f:np.savez_compressed(f,**{k:v[0].cpu().numpy() for k,v in outputs.items()})
    print(f"Saved base, anchor and final metric-depth maps at the 518x518 evaluation grid: {target}")
if __name__=="__main__":main()
