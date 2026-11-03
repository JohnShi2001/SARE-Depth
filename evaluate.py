"""Evaluate released paired checkpoints on a fixed CitrusFarm image manifest."""
import argparse
import csv
import json
from pathlib import Path
import torch
from sare_depth.model import SAREDepth, HeadOnlyDepth
from sare_depth.data import image_tensor, depth_tensor, read_manifest
from sare_depth.metrics import sufficient_statistics, aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/example.csv")
    parser.add_argument("--data-root", default="example_data")
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--seed", type=int, choices=[20260711,20260712,20260713], default=20260711)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/example")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--head-only", action="store_true", help="Evaluate the selected complete DPTHead control")
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Use a new output directory to retain previous evaluations")
    rows = read_manifest(args.manifest, args.data_root)
    model = (HeadOnlyDepth if args.head_only else SAREDepth)(args.weights, args.seed).to(args.device).eval()
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    records = {name: [] for name in (["head_only"] if args.head_only else ["base", "anchor", "final"])}
    with torch.inference_mode():
        for r in rows:
            outputs = model(image_tensor(r["image_path"])[None].to(args.device))
            gt = depth_tensor(r["depth_path"])
            for name, tensor in outputs.items():
                records[name].append({"sample_id":r["sample_id"], **sufficient_statistics(tensor[0].cpu(), gt)})
    out.mkdir(parents=True)
    for name, values in records.items():
        with (out / f"{name}_per_image.csv").open("w", newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    result={"protocol":"FP32; common cap30; GT-only finite [0.3,30] support; pixel pooled", "device":args.device,"seed":args.seed,"metrics":{name:aggregate(values) for name,values in records.items()}}
    (out / "summary.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()
