"""Portable inference entry points for the two bounded Round-2 bridges."""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

import round2_controls as core
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sare_depth.model import SAREDepth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('control', choices=['public-area', 'matched-reference'])
    args = parser.parse_args()
    dest = core.OUT / 'results' / args.control
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    core.seed(20260711)
    ds = core.Images('test')
    if args.control == 'public-area':
        model, identity = core.load('public')
        names = ['public-square-area']
    else:
        model = SAREDepth().cuda().eval()
        identity = {'assembly': 'configs/weights.json', 'LogLS': 0.16490098480565277,
                    'MetricLS': 0.15810321166638114, 'scale_fit': 'retained training-only coefficients'}
        names = ['base', 'LogLS', 'MetricLS', 'DP-RHR']
    records = {name: [] for name in names}
    with torch.inference_mode():
        for i in range(len(ds)):
            x, gt, sid = ds[i]
            if args.control == 'public-area':
                raw = cv2.imread(str(core.resolve(ds.rows[i]['image_path'])))
                if raw is None:
                    raise ValueError(ds.rows[i]['image_path'])
                square = cv2.resize(raw, (518, 518), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                x = (torch.from_numpy(rgb).permute(2, 0, 1) - core.MEAN) / core.STD
                predictions = {names[0]: core.depth(model, x[None].cuda())[0].cpu()}
            else:
                out = model.anchor(x[None].cuda())
                base = out['depth_pre_dp'][0].cpu()
                predictions = {'base': base, 'LogLS': base * identity['LogLS'],
                               'MetricLS': base * identity['MetricLS'],
                               'DP-RHR': out['depth'][0].cpu()}
            for name, pred in predictions.items():
                records[name].append({'sample_id': sid, **core.metrics(pred, gt)})
    result = {name: core.aggregate(rows) for name, rows in records.items()}
    assert len({r['valid_pixels'] for r in result.values()}) == 1
    for name, rows in records.items():
        core.table(dest / (name + '_per_image.csv'), rows)
    core.put(dest / 'summary.json', {'groups': result, 'identity': identity,
                                   'completed': core.now(), 'new_training_or_fitting': False})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
