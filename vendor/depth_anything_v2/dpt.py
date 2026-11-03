import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        return_features=False,
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken
        self.return_features = bool(return_features)

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def _build_feature_bundle(self, head_feature, decoder_multiscale):
        return {
            "head_feature": head_feature,
            "decoder_multiscale": decoder_multiscale,
        }

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        head_feature = self.scratch.output_conv1(path_1)
        out = F.interpolate(head_feature, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        if self.return_features:
            feature_bundle = self._build_feature_bundle(head_feature, (path_1, path_2, path_3, path_4))
            return out, feature_bundle
        return out


class MLPDepthHead(DPTHead):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        eps=1e-3,
        return_features=False,
    ):
        super().__init__(
            in_channels,
            features=features,
            use_bn=use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
            return_features=return_features,
        )
        hidden_channels = features // 2
        self.eps = float(eps)
        self.depth_mlp = nn.Sequential(
            nn.Conv2d(hidden_channels, 128, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        head_feature = self.scratch.output_conv1(path_1)
        out = F.interpolate(head_feature, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.depth_mlp(out)
        out = F.softplus(out) + self.eps
        out = out / (1.0 + out)

        if self.return_features:
            feature_bundle = self._build_feature_bundle(head_feature, (path_1, path_2, path_3, path_4))
            return out, feature_bundle
        return out


class ResidualMLPDepthCalibrator(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, num_layers=2):
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_layers = max(int(num_layers), 1)
        layers = []
        current_channels = in_channels
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(current_channels, hidden_dim, kernel_size=1, stride=1, padding=0))
            layers.append(nn.GELU())
            current_channels = hidden_dim
        layers.append(nn.Conv2d(current_channels, 1, kernel_size=1, stride=1, padding=0))
        self.mlp = nn.Sequential(*layers)

    def forward(self, feature_bundle):
        feature_map = feature_bundle["head_feature"] if isinstance(feature_bundle, dict) else feature_bundle
        return self.mlp(feature_map)


def compose_log_depth_residual(depth_base, residual, alpha, eps=1e-6, max_depth=None):
    if depth_base.shape != residual.shape:
        raise ValueError(
            f"depth_base shape {tuple(depth_base.shape)} does not match residual shape {tuple(residual.shape)}"
        )
    log_depth_base = torch.log(depth_base.clamp_min(eps))
    # Keep identity output at alpha=0 while preserving nonzero gradients for the residual branch.
    residual_gate = alpha * residual + (1.0 - alpha.detach()) * (residual - residual.detach())
    depth_final = torch.exp(log_depth_base + residual_gate)
    if max_depth is not None:
        depth_final = depth_final.clamp(min=eps, max=float(max_depth))
    else:
        depth_final = depth_final.clamp_min(eps)
    return depth_final, residual_gate


class DPMRHead(nn.Module):
    """Depth-preserving modular refinement head.

    The module keeps the original DPT depth prediction as D_base and learns only
    a bounded log-depth residual:

        log(D_final) = log(D_base) + alpha_max * g(z) * tanh(r(F))

    It is initialized as an identity mapping because the last local residual
    layer is zero initialized. The gate is intentionally initialized small.
    """

    def __init__(self, in_channels, hidden_dim=96, alpha_max=0.3, gate_bias_init=-3.0, eps=1e-6):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.alpha_max = float(alpha_max)
        self.eps = float(eps)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.local_residual = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1, padding=0),
        )
        context_dim = int(in_channels) * 2 + 4
        self.gate_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        final_conv = self.local_residual[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)
        final_linear = self.gate_mlp[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.constant_(final_linear.bias, float(gate_bias_init))

    def _depth_context(self, depth_base):
        depth = depth_base.clamp_min(self.eps)
        flat = depth.flatten(1)
        mean = flat.mean(dim=1, keepdim=True)
        std = (flat - mean).pow(2).mean(dim=1, keepdim=True).sqrt()
        log_depth = torch.log(depth)
        log_flat = log_depth.flatten(1)
        log_mean = log_flat.mean(dim=1, keepdim=True)
        log_std = (log_flat - log_mean).pow(2).mean(dim=1, keepdim=True).sqrt()
        return torch.cat([mean, std, log_mean, log_std], dim=1)

    def forward(self, feature_bundle, depth_base, max_depth=None):
        if not isinstance(feature_bundle, dict) or "head_feature" not in feature_bundle:
            raise KeyError("DPMRHead requires feature_bundle['head_feature']")
        feature_map = feature_bundle["head_feature"]
        residual_raw = self.local_residual(feature_map)
        if residual_raw.shape[-2:] != depth_base.shape[-2:]:
            residual_raw = F.interpolate(
                residual_raw,
                size=depth_base.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        residual_tanh = torch.tanh(residual_raw)
        pooled = torch.cat(
            [
                self.avg_pool(feature_map).flatten(1),
                self.max_pool(feature_map).flatten(1),
                self._depth_context(depth_base),
            ],
            dim=1,
        )
        gate = torch.sigmoid(self.gate_mlp(pooled)).view(-1, 1, 1, 1)
        log_delta = self.alpha_max * gate * residual_tanh
        depth_final = torch.exp(torch.log(depth_base.clamp_min(self.eps)) + log_delta)
        if max_depth is not None:
            depth_final = depth_final.clamp(min=self.eps, max=float(max_depth))
        else:
            depth_final = depth_final.clamp_min(self.eps)
        return depth_final, {
            "residual_raw": residual_raw,
            "residual_tanh": residual_tanh,
            "gate": gate,
            "log_delta": log_delta,
        }


class DPMRHeadV2(nn.Module):
    """Global-scale plus residual depth-preserving refinement head.

    This Phase15-B module separates global scale correction from local residual
    refinement:

        log(D_final) = log(D_base) + beta(z) + alpha_max * g(z) * tanh(r(F))

    The beta branch is bounded by beta_max, while the local residual branch is
    bounded by alpha_max. Both branches initialize close to identity.
    """

    def __init__(
        self,
        in_channels,
        hidden_dim=96,
        alpha_max=0.3,
        beta_max=2.0,
        gate_bias_init=-2.0,
        eps=1e-6,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.alpha_max = float(alpha_max)
        self.beta_max = float(beta_max)
        self.eps = float(eps)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.local_residual = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1, padding=0),
        )
        context_dim = int(in_channels) * 2 + 4
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
        )
        self.beta_head = nn.Linear(hidden_dim, 1)
        self.gate_head = nn.Linear(hidden_dim, 1)

        final_conv = self.local_residual[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(gate_bias_init))

    def _depth_context(self, depth_base):
        depth = depth_base.clamp_min(self.eps)
        flat = depth.flatten(1)
        mean = flat.mean(dim=1, keepdim=True)
        std = (flat - mean).pow(2).mean(dim=1, keepdim=True).sqrt()
        log_depth = torch.log(depth)
        log_flat = log_depth.flatten(1)
        log_mean = log_flat.mean(dim=1, keepdim=True)
        log_std = (log_flat - log_mean).pow(2).mean(dim=1, keepdim=True).sqrt()
        return torch.cat([log_mean, log_std, mean, std], dim=1)

    def forward(self, feature_bundle, depth_base, max_depth=None):
        if not isinstance(feature_bundle, dict) or "head_feature" not in feature_bundle:
            raise KeyError("DPMRHeadV2 requires feature_bundle['head_feature']")
        feature_map = feature_bundle["head_feature"]
        residual_raw = self.local_residual(feature_map)
        if residual_raw.shape[-2:] != depth_base.shape[-2:]:
            residual_raw = F.interpolate(
                residual_raw,
                size=depth_base.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        residual_tanh = torch.tanh(residual_raw)
        context = torch.cat(
            [
                self.avg_pool(feature_map).flatten(1),
                self.max_pool(feature_map).flatten(1),
                self._depth_context(depth_base),
            ],
            dim=1,
        )
        context_hidden = self.context_mlp(context)
        beta_raw = self.beta_head(context_hidden)
        beta = self.beta_max * torch.tanh(beta_raw)
        gate = torch.sigmoid(self.gate_head(context_hidden)).view(-1, 1, 1, 1)
        residual_delta = self.alpha_max * gate * residual_tanh
        beta_map = beta.view(-1, 1, 1, 1)
        total_log_delta = beta_map + residual_delta
        depth_final = torch.exp(torch.log(depth_base.clamp_min(self.eps)) + total_log_delta)
        if max_depth is not None:
            depth_final = depth_final.clamp(min=self.eps, max=float(max_depth))
        else:
            depth_final = depth_final.clamp_min(self.eps)
        return depth_final, {
            "beta_raw": beta_raw,
            "beta": beta,
            "residual_raw": residual_raw,
            "residual_tanh": residual_tanh,
            "gate": gate,
            "residual_delta": residual_delta,
            "total_log_delta": total_log_delta,
        }


class DPRHRSSFHead(nn.Module):
    """SSF-enhanced DP-RHR head for Phase16 fusion experiments."""

    def __init__(
        self,
        in_channels,
        hidden_dim=96,
        alpha_max=0.3,
        beta_max=2.0,
        gate_bias_init=-2.0,
        gamma_max_delta=0.5,
        shift_max=0.5,
        eps=1e-6,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.alpha_max = float(alpha_max)
        self.beta_max = float(beta_max)
        self.gamma_max_delta = float(gamma_max_delta)
        self.shift_max = float(shift_max)
        self.eps = float(eps)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        context_dim = int(in_channels) * 2 + 4

        self.ssf_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
        )
        self.gamma_head = nn.Linear(hidden_dim, in_channels)
        self.shift_head = nn.Linear(hidden_dim, in_channels)

        self.local_residual = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1, padding=0),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
        )
        self.beta_head = nn.Linear(hidden_dim, 1)
        self.gate_head = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.shift_head.weight)
        nn.init.zeros_(self.shift_head.bias)
        final_conv = self.local_residual[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(gate_bias_init))

    def _depth_context(self, depth_base):
        depth = depth_base.clamp_min(self.eps)
        flat = depth.flatten(1)
        mean = flat.mean(dim=1, keepdim=True)
        std = (flat - mean).pow(2).mean(dim=1, keepdim=True).sqrt()
        log_depth = torch.log(depth)
        log_flat = log_depth.flatten(1)
        log_mean = log_flat.mean(dim=1, keepdim=True)
        log_std = (log_flat - log_mean).pow(2).mean(dim=1, keepdim=True).sqrt()
        return torch.cat([log_mean, log_std, mean, std], dim=1)

    def _context(self, feature_map, depth_base):
        return torch.cat(
            [
                self.avg_pool(feature_map).flatten(1),
                self.max_pool(feature_map).flatten(1),
                self._depth_context(depth_base),
            ],
            dim=1,
        )

    def forward(self, feature_bundle, depth_base, max_depth=None):
        if not isinstance(feature_bundle, dict) or "head_feature" not in feature_bundle:
            raise KeyError("DPRHRSSFHead requires feature_bundle['head_feature']")
        feature_map = feature_bundle["head_feature"]
        ssf_hidden = self.ssf_mlp(self._context(feature_map, depth_base))
        gamma_raw = self.gamma_head(ssf_hidden)
        shift_raw = self.shift_head(ssf_hidden)
        gamma = 1.0 + self.gamma_max_delta * torch.tanh(gamma_raw)
        shift = self.shift_max * torch.tanh(shift_raw)
        feature_ssf = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1) * feature_map + shift.view(
            shift.shape[0], shift.shape[1], 1, 1
        )

        residual_raw = self.local_residual(feature_ssf)
        if residual_raw.shape[-2:] != depth_base.shape[-2:]:
            residual_raw = F.interpolate(
                residual_raw,
                size=depth_base.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        residual_tanh = torch.tanh(residual_raw)
        context_hidden = self.context_mlp(self._context(feature_ssf, depth_base))
        beta_raw = self.beta_head(context_hidden)
        beta = self.beta_max * torch.tanh(beta_raw)
        gate = torch.sigmoid(self.gate_head(context_hidden)).view(-1, 1, 1, 1)
        residual_delta = self.alpha_max * gate * residual_tanh
        beta_map = beta.view(-1, 1, 1, 1)
        total_log_delta = beta_map + residual_delta
        depth_final = torch.exp(torch.log(depth_base.clamp_min(self.eps)) + total_log_delta)
        if max_depth is not None:
            depth_final = depth_final.clamp(min=self.eps, max=float(max_depth))
        else:
            depth_final = depth_final.clamp_min(self.eps)
        return depth_final, {
            "beta_raw": beta_raw,
            "beta": beta,
            "residual_raw": residual_raw,
            "residual_tanh": residual_tanh,
            "gate": gate,
            "residual_delta": residual_delta,
            "total_log_delta": total_log_delta,
            "gamma_raw": gamma_raw,
            "gamma": gamma,
            "shift_raw": shift_raw,
            "shift": shift,
        }


class SingleScaleMLPFocalHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, eps=1e-6, predict_fov=False):
        super().__init__()
        self.eps = float(eps)
        self.predict_fov = bool(predict_fov)
        hidden_dim = int(hidden_dim)
        mid_dim = max(hidden_dim // 2, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 1),
        )

    def _compute_fov(self, focal_px, image_width):
        if image_width is None:
            return None
        width = image_width.to(focal_px.dtype).clamp_min(self.eps)
        return 2.0 * torch.rad2deg(torch.atan(0.5 * width / focal_px.clamp_min(self.eps)))

    def forward(self, feature_bundle, image_hw=None):
        feature_map = feature_bundle["head_feature"] if isinstance(feature_bundle, dict) else feature_bundle
        pooled = self.pool(feature_map).flatten(1)
        log_focal = self.mlp(pooled)
        focal_px = torch.exp(log_focal).clamp_min(self.eps)
        outputs = {
            "student_log_focal": log_focal,
            "student_focal_px": focal_px,
        }
        if self.predict_fov and image_hw is not None:
            image_width = image_hw[:, 1:2]
            outputs["student_fov_deg"] = self._compute_fov(focal_px, image_width)
        return outputs


class MultiScalePooledMLPFocalHead(nn.Module):
    def __init__(self, in_channels, num_scales=4, hidden_dim=192, eps=1e-6, predict_fov=False):
        super().__init__()
        self.eps = float(eps)
        self.predict_fov = bool(predict_fov)
        self.num_scales = int(num_scales)
        input_dim = self.num_scales * (2 * in_channels) + 3
        mid_dim = max(hidden_dim // 2, 64)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 1),
        )

    def _encode_image_hw(self, image_hw, device, dtype):
        if image_hw is None:
            return torch.zeros((1, 3), device=device, dtype=dtype)
        h = image_hw[:, 0:1].to(dtype).clamp_min(1.0)
        w = image_hw[:, 1:2].to(dtype).clamp_min(1.0)
        return torch.cat([torch.log(h), torch.log(w), torch.log(w / h)], dim=1)

    def _compute_fov(self, focal_px, image_width):
        if image_width is None:
            return None
        width = image_width.to(focal_px.dtype).clamp_min(self.eps)
        return 2.0 * torch.rad2deg(torch.atan(0.5 * width / focal_px.clamp_min(self.eps)))

    def forward(self, feature_bundle, image_hw=None):
        if not isinstance(feature_bundle, dict) or "decoder_multiscale" not in feature_bundle:
            raise KeyError("MultiScalePooledMLPFocalHead requires feature_bundle['decoder_multiscale']")
        features = feature_bundle["decoder_multiscale"]
        pooled_features = []
        for feat in features:
            pooled_features.append(self.avg_pool(feat).flatten(1))
            pooled_features.append(self.max_pool(feat).flatten(1))
        image_encoding = self._encode_image_hw(image_hw, features[0].device, features[0].dtype)
        if image_encoding.shape[0] == 1 and pooled_features[0].shape[0] != 1:
            image_encoding = image_encoding.expand(pooled_features[0].shape[0], -1)
        fused = torch.cat(pooled_features + [image_encoding], dim=1)
        log_focal = self.mlp(fused)
        focal_px = torch.exp(log_focal).clamp_min(self.eps)
        outputs = {
            "student_log_focal": log_focal,
            "student_focal_px": focal_px,
        }
        if self.predict_fov and image_hw is not None:
            image_width = image_hw[:, 1:2]
            outputs["student_fov_deg"] = self._compute_fov(focal_px, image_width)
        return outputs


class FocalFoVHead(SingleScaleMLPFocalHead):
    def __init__(self, in_channels, hidden_dim=128, predict_fov=False, eps=1e-6):
        super().__init__(in_channels=in_channels, hidden_dim=hidden_dim, eps=eps, predict_fov=predict_fov)


class DepthAnythingV2(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        max_depth=20.0,
        depth_head_type="original",
        enable_mlp_depth_head=False,
        enable_mlp_residual=False,
        residual_mlp_hidden_dim=128,
        residual_mlp_num_layers=2,
        residual_alpha_init=0.0,
        enable_dp_rhr=False,
        dp_rhr_hidden_dim=96,
        dp_rhr_alpha_max=0.3,
        dp_rhr_gate_bias_init=-3.0,
        enable_dp_rhr_v2=False,
        dp_rhr_v2_hidden_dim=96,
        dp_rhr_v2_alpha_max=0.3,
        dp_rhr_v2_beta_max=2.0,
        dp_rhr_v2_gate_bias_init=-2.0,
        enable_dprhr_ssf=False,
        dprhr_ssf_hidden_dim=96,
        dprhr_ssf_alpha_max=0.3,
        dprhr_ssf_beta_max=2.0,
        dprhr_ssf_gate_bias_init=-2.0,
        dprhr_ssf_gamma_max_delta=0.5,
        dprhr_ssf_shift_max=0.5,
        enable_focal_head=False,
        focal_head_type="none",
        focal_output="focal_px",
    ):
        super(DepthAnythingV2, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }

        self.max_depth = max_depth

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        requested_head = str(depth_head_type).strip().lower() if depth_head_type is not None else "original"
        use_mlp_depth_head = bool(enable_mlp_depth_head) or requested_head in {"mlp", "dav2_mlp_depth_head"}
        self.depth_head_type = "mlp" if use_mlp_depth_head else "original"
        self.enable_mlp_residual = bool(enable_mlp_residual)
        self.enable_dp_rhr = bool(enable_dp_rhr)
        self.enable_dp_rhr_v2 = bool(enable_dp_rhr_v2)
        self.enable_dprhr_ssf = bool(enable_dprhr_ssf)
        self.enable_focal_head = bool(enable_focal_head)
        self.focal_head_type = str(focal_head_type).strip().lower() if focal_head_type is not None else "none"
        self.focal_output = str(focal_output).strip().lower() if focal_output is not None else "focal_px"

        head_cls = MLPDepthHead if use_mlp_depth_head else DPTHead
        self.depth_head = head_cls(
            self.pretrained.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
            return_features=self.enable_focal_head
            or self.enable_mlp_residual
            or self.enable_dp_rhr
            or self.enable_dp_rhr_v2
            or self.enable_dprhr_ssf,
        )

        self.residual_mlp = None
        self.residual_alpha = None
        if self.enable_mlp_residual:
            self.residual_mlp = ResidualMLPDepthCalibrator(
                features // 2,
                hidden_dim=residual_mlp_hidden_dim,
                num_layers=residual_mlp_num_layers,
            )
            self.residual_alpha = nn.Parameter(torch.tensor(float(residual_alpha_init), dtype=torch.float32))

        self.dp_rhr = None
        if self.enable_dp_rhr:
            self.dp_rhr = DPMRHead(
                features // 2,
                hidden_dim=dp_rhr_hidden_dim,
                alpha_max=dp_rhr_alpha_max,
                gate_bias_init=dp_rhr_gate_bias_init,
                eps=1e-6,
            )

        self.dp_rhr_v2 = None
        if self.enable_dp_rhr_v2:
            self.dp_rhr_v2 = DPMRHeadV2(
                features // 2,
                hidden_dim=dp_rhr_v2_hidden_dim,
                alpha_max=dp_rhr_v2_alpha_max,
                beta_max=dp_rhr_v2_beta_max,
                gate_bias_init=dp_rhr_v2_gate_bias_init,
                eps=1e-6,
            )

        self.dprhr_ssf = None
        if self.enable_dprhr_ssf:
            self.dprhr_ssf = DPRHRSSFHead(
                features // 2,
                hidden_dim=dprhr_ssf_hidden_dim,
                alpha_max=dprhr_ssf_alpha_max,
                beta_max=dprhr_ssf_beta_max,
                gate_bias_init=dprhr_ssf_gate_bias_init,
                gamma_max_delta=dprhr_ssf_gamma_max_delta,
                shift_max=dprhr_ssf_shift_max,
                eps=1e-6,
            )

        self.focal_head = None
        if self.enable_focal_head:
            focal_head_key = self.focal_head_type or "none"
            predict_fov = self.focal_output == "fov_deg"
            if focal_head_key in {"mlp", "single_scale_mlp", "df_a"}:
                self.focal_head = SingleScaleMLPFocalHead(features // 2, hidden_dim=128, predict_fov=predict_fov)
            elif focal_head_key in {"multi_scale_pooled_mlp", "df_b"}:
                self.focal_head = MultiScalePooledMLPFocalHead(features, num_scales=4, hidden_dim=192, predict_fov=predict_fov)
            else:
                raise ValueError(f"Unsupported focal_head_type: {self.focal_head_type}")

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)

        if self.enable_focal_head or self.enable_mlp_residual or self.enable_dp_rhr or self.enable_dp_rhr_v2 or self.enable_dprhr_ssf:
            depth_pred = self.depth_head(features, patch_h, patch_w)
            if isinstance(depth_pred, tuple):
                depth_map, feature_bundle = depth_pred
            else:
                raise RuntimeError("Auxiliary heads require return_features=True depth head outputs")

            depth_base = depth_map * self.max_depth
            depth = depth_base
            outputs = {}
            if self.enable_mlp_residual and self.residual_mlp is not None and self.residual_alpha is not None:
                residual = self.residual_mlp(feature_bundle)
                if residual.shape[-2:] != depth_base.shape[-2:]:
                    residual = F.interpolate(
                        residual,
                        size=depth_base.shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                depth, residual_gate = compose_log_depth_residual(
                    depth_base,
                    residual,
                    self.residual_alpha,
                    eps=1e-6,
                    max_depth=self.max_depth,
                )
                outputs.update(
                    {
                        "depth_base": depth_base.squeeze(1),
                        "depth_residual": residual.squeeze(1),
                        "depth_residual_gate": residual_gate.squeeze(1),
                        "residual_alpha": self.residual_alpha.reshape(1),
                    }
                )

            if self.enable_dp_rhr and self.dp_rhr is not None:
                dpmr_depth, dpmr_info = self.dp_rhr(
                    feature_bundle,
                    depth,
                    max_depth=self.max_depth,
                )
                depth = dpmr_depth
                outputs.update(
                    {
                        "depth_base": depth_base.squeeze(1),
                        "dp_rhr_residual_raw": dpmr_info["residual_raw"].squeeze(1),
                        "dp_rhr_residual_tanh": dpmr_info["residual_tanh"].squeeze(1),
                        "dp_rhr_log_delta": dpmr_info["log_delta"].squeeze(1),
                        "dp_rhr_gate": dpmr_info["gate"].reshape(-1),
                        "dp_rhr_alpha_max": torch.full(
                            (depth.shape[0],),
                            float(self.dp_rhr.alpha_max),
                            device=depth.device,
                            dtype=depth.dtype,
                        ),
                    }
                )

            if self.enable_dp_rhr_v2 and self.dp_rhr_v2 is not None:
                dpmr_depth, dpmr_info = self.dp_rhr_v2(
                    feature_bundle,
                    depth,
                    max_depth=self.max_depth,
                )
                depth = dpmr_depth
                outputs.update(
                    {
                        "depth_base": depth_base.squeeze(1),
                        "dp_rhr_v2_beta": dpmr_info["beta"].reshape(-1),
                        "dp_rhr_v2_beta_raw": dpmr_info["beta_raw"].reshape(-1),
                        "dp_rhr_v2_residual_raw": dpmr_info["residual_raw"].squeeze(1),
                        "dp_rhr_v2_residual_tanh": dpmr_info["residual_tanh"].squeeze(1),
                        "dp_rhr_v2_residual_delta": dpmr_info["residual_delta"].squeeze(1),
                        "dp_rhr_v2_total_log_delta": dpmr_info["total_log_delta"].squeeze(1),
                        "dp_rhr_v2_gate": dpmr_info["gate"].reshape(-1),
                        "dp_rhr_v2_alpha_max": torch.full(
                            (depth.shape[0],),
                            float(self.dp_rhr_v2.alpha_max),
                            device=depth.device,
                            dtype=depth.dtype,
                        ),
                        "dp_rhr_v2_beta_max": torch.full(
                            (depth.shape[0],),
                            float(self.dp_rhr_v2.beta_max),
                            device=depth.device,
                            dtype=depth.dtype,
                        ),
                    }
                )

            if self.enable_dprhr_ssf and self.dprhr_ssf is not None:
                dprhr_ssf_depth, dprhr_ssf_info = self.dprhr_ssf(
                    feature_bundle,
                    depth,
                    max_depth=self.max_depth,
                )
                depth = dprhr_ssf_depth
                outputs.update(
                    {
                        "depth_base": depth_base.squeeze(1),
                        "dprhr_ssf_beta": dprhr_ssf_info["beta"].reshape(-1),
                        "dprhr_ssf_beta_raw": dprhr_ssf_info["beta_raw"].reshape(-1),
                        "dprhr_ssf_residual_raw": dprhr_ssf_info["residual_raw"].squeeze(1),
                        "dprhr_ssf_residual_tanh": dprhr_ssf_info["residual_tanh"].squeeze(1),
                        "dprhr_ssf_residual_delta": dprhr_ssf_info["residual_delta"].squeeze(1),
                        "dprhr_ssf_total_log_delta": dprhr_ssf_info["total_log_delta"].squeeze(1),
                        "dprhr_ssf_gate": dprhr_ssf_info["gate"].reshape(-1),
                        "dprhr_ssf_gamma": dprhr_ssf_info["gamma"],
                        "dprhr_ssf_gamma_raw": dprhr_ssf_info["gamma_raw"],
                        "dprhr_ssf_shift": dprhr_ssf_info["shift"],
                        "dprhr_ssf_shift_raw": dprhr_ssf_info["shift_raw"],
                        "dprhr_ssf_alpha_max": torch.full(
                            (depth.shape[0],),
                            float(self.dprhr_ssf.alpha_max),
                            device=depth.device,
                            dtype=depth.dtype,
                        ),
                        "dprhr_ssf_beta_max": torch.full(
                            (depth.shape[0],),
                            float(self.dprhr_ssf.beta_max),
                            device=depth.device,
                            dtype=depth.dtype,
                        ),
                    }
                )

            if self.enable_focal_head and self.focal_head is not None:
                batch_size = x.shape[0]
                image_hw = torch.tensor([x.shape[-2], x.shape[-1]], device=x.device, dtype=depth.dtype).view(1, 2).repeat(batch_size, 1)
                outputs.update(self.focal_head(feature_bundle, image_hw=image_hw))

            outputs["depth"] = depth.squeeze(1)
            return outputs

        depth = self.depth_head(features, patch_h, patch_w) * self.max_depth

        return depth.squeeze(1)

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)

        depth = self.forward(image)
        if isinstance(depth, dict):
            depth = depth.get("depth")
            if depth is None:
                raise KeyError("Model forward returned a dict without a 'depth' key")

        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]

        return depth.cpu().numpy()

    def image2tensor(self, raw_image, input_size=518):
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        h, w = raw_image.shape[:2]

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)

        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)

        return image, (h, w)
