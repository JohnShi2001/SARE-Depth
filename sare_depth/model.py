from pathlib import Path
import hashlib
import json
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from depth_anything_v2.dpt import DepthAnythingV2
from .historical_heads import DPAnchorInterface, ErrorCalibratedRangeResidualHead


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_verified(weights, name):
    manifest = json.loads((ROOT / "configs/weights.json").read_text())
    entry = next(r for r in manifest if Path(r["file"]).name == name)
    path = Path(weights) / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run python download_weights.py first.")
    if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
        raise ValueError(f"Checkpoint integrity check failed: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a tensor-only state dictionary")
    return state


class SAREDepth(torch.nn.Module):
    def __init__(self, weights=ROOT / "weights", seed=20260711):
        super().__init__()
        model = DepthAnythingV2(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384],
            max_depth=80.0, enable_focal_head=True, focal_head_type="df_b",
            enable_dp_rhr_v2=True, dp_rhr_v2_hidden_dim=96,
            dp_rhr_v2_alpha_max=0.3, dp_rhr_v2_beta_max=2.0,
            dp_rhr_v2_gate_bias_init=-2.0,
        )
        state = load_verified(weights, "stage1_base.pth")
        adapter = load_verified(weights, "dprhr_anchor.pth")
        state.update({"dp_rhr_v2." + k: v for k, v in adapter.items()})
        model.load_state_dict(state, strict=True)
        self.anchor = DPAnchorInterface(model)
        self.refinement = ErrorCalibratedRangeResidualHead(
            use_aep=False, use_range_experts=True, alpha=0.2,
            initial_temperature=0.35, max_depth=30.0,
        )
        self.refinement.load_state_dict(load_verified(weights, f"re_historical_{seed}.pth"), strict=True)
        self.requires_grad_(False)
        self.eval()

    def forward(self, image):
        a = self.anchor(image)
        r = self.refinement(a["depth"], a["head_feature"])
        return {"base": a["depth_pre_dp"], "anchor": a["depth"], "final": r["depth"]}


class HeadOnlyDepth(torch.nn.Module):
    """The retained base encoder with the selected complete original DPTHead."""
    def __init__(self, weights=ROOT / "weights", seed=20260711):
        super().__init__()
        self.model = DepthAnythingV2(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384],
            max_depth=80.0, enable_focal_head=True, focal_head_type="df_b",
        )
        self.model.load_state_dict(load_verified(weights, "stage1_base.pth"), strict=True)
        self.model.depth_head.load_state_dict(load_verified(weights, f"head_only_{seed}.pth"), strict=True)
        self.requires_grad_(False)
        self.eval()

    def forward(self, image):
        return {"head_only": self.model(image)["depth"]}
