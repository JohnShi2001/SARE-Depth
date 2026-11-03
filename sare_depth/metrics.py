import math
import torch


def sufficient_statistics(prediction, reference, cap=30.0):
    support = torch.isfinite(reference) & (reference >= 0.3) & (reference <= 30.0)
    p = prediction[support].double()
    g = reference[support].double()
    if not len(g):
        raise ValueError("Empty reference support")
    invalid = ~torch.isfinite(p) | (p <= 0)
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} invalid predictions on fixed reference support")
    cap_hits = int((p > cap).sum()) if cap is not None else 0
    if cap is not None:
        p = p.clamp(1e-6, cap)
    e = p - g
    ratio = torch.maximum(p / g, g / p)
    return {
        "pixels": len(g), "sum_absrel": float((e.abs() / g).sum()),
        "sum_sq": float(e.square().sum()), "sum_abs": float(e.abs().sum()),
        "sum_log10": float((p.log10() - g.log10()).abs().sum()),
        "delta1_count": int((ratio < 1.25).sum()),
        "delta2_count": int((ratio < 1.25 ** 2).sum()),
        "delta3_count": int((ratio < 1.25 ** 3).sum()), "cap_hits": cap_hits,
    }


def aggregate(rows):
    n = sum(r["pixels"] for r in rows)
    if not n:
        raise ValueError("No evaluated pixels")
    total = lambda key: sum(r[key] for r in rows)
    return {
        "images": len(rows), "valid_pixels": n,
        "AbsRel": total("sum_absrel") / n, "RMSE_m": math.sqrt(total("sum_sq") / n),
        "MAE_m": total("sum_abs") / n, "log10": total("sum_log10") / n,
        **{f"delta{k}": total(f"delta{k}_count") / n for k in [1, 2, 3]},
        "above_cap_pixels": total("cap_hits"),
    }
