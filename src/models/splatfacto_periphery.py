"""exp022: periphery-region-weighted L1 loss (from exp020 finding (a), gated by
exp021's verdict that the frame-corner error is TOP-biased overlap/frustum, not
symmetric lens distortion).

exp013 up-weighted whole *train images* that cover test poses. exp022 instead
up-weights the loss *within* each image, in the region exp020/exp021 measured as
chronically under-fit: the top edge and (more so) the top corners. A per-pixel
weight map multiplies the L1 term only (the top-biased weakness is a PSNR/detail
one -- exp020 finding (b): GT sharpness -> PSNR r=-0.54, -> LPIPS r~0 -- so SSIM
stays global). The map is mean-normalized to 1.0, so at uniform error the loss is
byte-identical to baseline and learning-rate/scale comparisons stay valid;
boost=0 reproduces stock splatfacto exactly (used as the A/B control).

Registered as `splatfacto-prw` in src/register_custom_methods.py. Pilot on
hcm0034 only (plan_execution_v3 ledger exp022).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Type

import torch
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig


class PeripheryWeightedLossMixin:
    """Replaces the uniform-mean L1 with a top/corner-weighted, mean-1 L1."""

    def _periphery_weight(self, H: int, W: int, device) -> torch.Tensor:
        cache = getattr(self, "_prw_cache", None)
        if cache is not None and cache[0] == (H, W):
            return cache[1]
        cfg = self.config
        vy = torch.linspace(0.0, 1.0, H, device=device)          # 0 = top row
        vx = torch.linspace(0.0, 1.0, W, device=device)
        top_frac = max(cfg.prw_top_frac, 1e-3)
        top = (1.0 - vy / top_frac).clamp(0.0, 1.0)[:, None]      # [H,1] top ramp
        edge_x = (2.0 * vx - 1.0).abs()[None, :]                  # [1,W] 0 center..1 sides
        emphasis = top * (1.0 + cfg.prw_corner_gain * edge_x)    # [H,W] top + corner bias
        m = emphasis.max()
        if m > 0:
            emphasis = emphasis / m                               # -> [0,1]
        wmap = 1.0 + cfg.prw_boost * emphasis
        wmap = wmap / wmap.mean()                                 # mean-1 (LR-neutral)
        self._prw_cache = ((H, W), wmap)
        return wmap

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        cfg = self.config
        if cfg.prw_boost <= 0 or (self.training and self.step < cfg.prw_start_step):
            return loss_dict  # boost=0 (control) or warmup: identical to baseline

        # Replicate the parent's gt/pred prep exactly (composite + optional mask).
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]
        if "mask" in batch:
            mask = self._downscale_if_required(batch["mask"]).to(self.device)
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        H, W = gt_img.shape[:2]
        wmap = self._periphery_weight(H, W, gt_img.device)
        per_pixel = torch.abs(gt_img - pred_img).mean(dim=-1)     # [H,W]
        Ll1 = (wmap * per_pixel).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None], pred_img.permute(2, 0, 1)[None])
        loss_dict["main_loss"] = (1 - cfg.ssim_lambda) * Ll1 + cfg.ssim_lambda * simloss
        return loss_dict


@dataclass
class SplatfactoPeripheryModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: SplatfactoPeripheryModel)
    prw_boost: float = 1.0
    """Peak extra L1 weight at the top corners before mean-normalization (0 = control)."""
    prw_top_frac: float = 0.5
    """Fraction of image height (from the top) over which the top emphasis ramps to 0."""
    prw_corner_gain: float = 1.0
    """Extra emphasis at the top-left/right corners vs the top-center (0 = uniform top band)."""
    prw_start_step: int = 0
    """Step to enable the weighting at (0 = from the start)."""


class SplatfactoPeripheryModel(PeripheryWeightedLossMixin, SplatfactoModel):
    config: SplatfactoPeripheryModelConfig
