from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPAnchorInterface(nn.Module):
    """Expose frozen DP-RHR-B depth and head features without changing its path."""

    def __init__(self, model: nn.Module):
        super().__init__()
        if model.dp_rhr_v2 is None or not model.enable_dp_rhr_v2:
            raise ValueError("DPAnchorInterface requires an enabled dp_rhr_v2 module")
        self.model = model
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def max_depth(self):
        return float(self.model.max_depth)

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        patch_h, patch_w = image.shape[-2] // 14, image.shape[-1] // 14
        features = self.model.pretrained.get_intermediate_layers(
            image,
            self.model.intermediate_layer_idx[self.model.encoder],
            return_class_token=True,
        )
        depth_pred = self.model.depth_head(features, patch_h, patch_w)
        if not isinstance(depth_pred, tuple):
            raise RuntimeError("DP-RHR-B depth head must return depth and feature bundle")
        depth_map, feature_bundle = depth_pred
        depth_pre_dp = depth_map * self.model.max_depth
        depth_b, info = self.model.dp_rhr_v2(feature_bundle, depth_pre_dp, max_depth=self.model.max_depth)
        return {
            "depth": depth_b.squeeze(1),
            "depth_pre_dp": depth_pre_dp.squeeze(1),
            "head_feature": feature_bundle["head_feature"],
            "camera_context": feature_bundle.get("camera_context"),
            "beta": info["beta"].reshape(-1),
            "beta_raw": info["beta_raw"].reshape(-1),
            "gate": info["gate"].reshape(-1),
            "residual_raw": info["residual_raw"].squeeze(1),
            "residual_tanh": info["residual_tanh"].squeeze(1),
            "residual_delta": info["residual_delta"].squeeze(1),
            "total_log_delta": info["total_log_delta"].squeeze(1),
        }

    def forward_static(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward_static_inputs(image)[0]

    def forward_static_inputs(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed-shape export route that avoids ONNX squeeze control flow."""
        patch_h, patch_w = image.shape[-2] // 14, image.shape[-1] // 14
        features = self.model.pretrained.get_intermediate_layers(
            image,
            self.model.intermediate_layer_idx[self.model.encoder],
            return_class_token=True,
        )
        depth_pred = self.model.depth_head(features, patch_h, patch_w)
        if not isinstance(depth_pred, tuple):
            raise RuntimeError("DP-RHR-B depth head must return depth and feature bundle")
        depth_map, feature_bundle = depth_pred
        depth_pre_dp = depth_map * self.model.max_depth
        depth_b, _ = self.model.dp_rhr_v2(feature_bundle, depth_pre_dp, max_depth=self.model.max_depth)
        # Explicitly index the guaranteed singleton channel instead of squeeze(1).
        return depth_b[:, 0, :, :], feature_bundle["head_feature"]


class AgriculturalErrorPredictor(nn.Module):
    """Predict normalized DP-RHR-B log-depth error from frozen features."""

    def __init__(self, feature_channels: int = 32, hidden_channels: int = 32):
        super().__init__()
        input_channels = int(feature_channels) + 1
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, feature: torch.Tensor, depth_b: torch.Tensor) -> torch.Tensor:
        log_depth = torch.log(depth_b.clamp_min(1.0e-6))[:, None]
        log_depth = F.interpolate(log_depth, size=feature.shape[-2:], mode="bilinear", align_corners=True)
        q = torch.sigmoid(self.net(torch.cat([feature, log_depth], dim=1)))
        return F.interpolate(q, size=depth_b.shape[-2:], mode="bilinear", align_corners=True)[:, 0]


class LightweightResidualTrunk(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)


class ErrorCalibratedRangeResidualHead(nn.Module):
    """AEP-gated single or soft range-conditioned residual refinement."""

    def __init__(
        self,
        use_aep: bool,
        use_range_experts: bool,
        aep_only: bool = False,
        channels: int = 32,
        alpha: float = 0.2,
        initial_temperature: float = 0.35,
        max_depth: float = 80.0,
    ):
        super().__init__()
        self.use_aep = bool(use_aep)
        self.use_range_experts = bool(use_range_experts)
        self.aep_only = bool(aep_only)
        self.alpha = float(alpha)
        self.max_depth = float(max_depth)
        self.aep = AgriculturalErrorPredictor(channels) if self.use_aep else None
        self.trunk = None if self.aep_only else LightweightResidualTrunk(channels)
        self.output_head = None
        self.expert_heads = None
        if not self.aep_only and self.use_range_experts:
            self.expert_heads = nn.ModuleList([nn.Conv2d(channels, 1, 1) for _ in range(4)])
            self.register_buffer("centers", torch.log(torch.tensor([1.0, 3.5, 7.5, 15.0], dtype=torch.float32)))
            normalized = (float(initial_temperature) - 0.05) / 0.95
            raw = math.log(normalized / (1.0 - normalized))
            self.register_buffer("temperature_raw", torch.full((4,), raw, dtype=torch.float32))
            for head in self.expert_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        elif not self.aep_only:
            self.output_head = nn.Conv2d(channels, 1, 1)
            nn.init.zeros_(self.output_head.weight)
            nn.init.zeros_(self.output_head.bias)

    @property
    def temperatures(self):
        if not self.use_range_experts:
            return None
        return 0.05 + 0.95 * torch.sigmoid(self.temperature_raw)

    def forward(self, depth_b: torch.Tensor, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        q = self.aep(feature, depth_b) if self.aep is not None else torch.ones_like(depth_b)
        if self.aep_only:
            return {"depth": depth_b, "depth_b": depth_b, "q": q, "residual_delta": torch.zeros_like(depth_b)}
        trunk = self.trunk(feature)
        range_weights = None
        expert_residual_raw = None
        if self.use_range_experts:
            log_depth = torch.log(depth_b.clamp_min(1.0e-6))
            logits = -(
                (log_depth[:, None] - self.centers[None, :, None, None]).pow(2)
                / self.temperatures[None, :, None, None]
            )
            range_weights = torch.softmax(logits, dim=1)
            expert_residual_raw = torch.cat(
                [F.interpolate(head(trunk), size=depth_b.shape[-2:], mode="bilinear", align_corners=True) for head in self.expert_heads],
                dim=1,
            )
            residual_raw = (range_weights * expert_residual_raw).sum(dim=1)
        else:
            residual_raw = F.interpolate(self.output_head(trunk), size=depth_b.shape[-2:], mode="bilinear", align_corners=True)[:, 0]
        residual_tanh = torch.tanh(residual_raw)
        gate = q if self.use_aep else torch.ones_like(q)
        residual_delta = self.alpha * gate * residual_tanh
        depth = torch.exp(torch.log(depth_b.clamp_min(1.0e-6)) + residual_delta).clamp(1.0e-6, self.max_depth)
        output = {
            "depth": depth,
            "depth_b": depth_b,
            "q": q,
            "residual_raw": residual_raw,
            "residual_tanh": residual_tanh,
            "residual_delta": residual_delta,
        }
        if range_weights is not None:
            output["range_weights"] = range_weights
            output["expert_residual_raw"] = expert_residual_raw
            output["temperatures"] = self.temperatures
        return output


class DPRHRECRHRModel(nn.Module):
    def __init__(self, anchor: DPAnchorInterface, adapter: ErrorCalibratedRangeResidualHead):
        super().__init__()
        self.anchor = anchor
        self.adapter = adapter
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.anchor.eval()
        return self

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            base = self.anchor(image)
        output = self.adapter(base["depth"], base["head_feature"])
        output.update({f"anchor_{key}": value for key, value in base.items() if key not in {"depth", "head_feature"}})
        return output

    def forward_static(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            depth_b, head_feature = self.anchor.forward_static_inputs(image)
        return self.adapter(depth_b, head_feature)["depth"]


def phase21_parameter_scope(model: DPRHRECRHRModel) -> dict[str, int]:
    return {
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "frozen": int(sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)),
        "dp_rhr_b_anchor": int(sum(parameter.numel() for parameter in model.anchor.parameters())),
        "new_module": int(sum(parameter.numel() for parameter in model.adapter.parameters())),
    }
