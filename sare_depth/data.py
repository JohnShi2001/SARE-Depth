from pathlib import Path
import csv
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def image_tensor(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((518, 518), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.array(rgb, dtype=np.float32) / 255).permute(2, 0, 1)
    return (tensor - MEAN) / STD


def depth_tensor(path):
    array = np.load(path, allow_pickle=False).astype(np.float32)
    if array.ndim == 3:
        if array.shape[-1] != 1:
            raise ValueError("Depth arrays must be HxW or HxWx1, in metres")
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError("Depth arrays must be two-dimensional")
    return F.interpolate(torch.from_numpy(array)[None, None], (518, 518), mode="nearest")[0, 0]


def read_manifest(path, root):
    root = Path(root).resolve()
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Duplicate sample IDs")
    for row in records:
        for field in ("image_path", "depth_path"):
            relative = Path(row[field])
            resolved = (root / relative).resolve()
            if relative.is_absolute() or not resolved.is_relative_to(root):
                raise ValueError("The public manifest must use paths within data-root")
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            row[field] = str(resolved)
    return records
