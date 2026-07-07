"""Splatfacto + LPIPS(VGG) loss term for the exp009 perceptual fine-tune.

Usage pattern (plan_execution_v3 Day 3): load the best Tier-A checkpoint and
run +5-10k extra iterations with the perceptual term added at weight
0.05-0.1. The LPIPS net is VGG (the *loss*-appropriate backbone per the
LPIPS paper) while local eval keeps alex -- do not conflate the two.

Implemented as a mixin so it composes with the MCMC densifier:
  splatfacto-perceptual       = PerceptualLossMixin + SplatfactoModel
  splatfacto-mcmc-perceptual  = PerceptualLossMixin + SplatfactoMCMCModel
(both registered in src/register_custom_methods.py).

VRAM guard: LPIPS(VGG) backward at full 1320x989 is heavy; by default the
perceptual term is computed on a random 512x512 crop per step
(lpips_patch_size=0 -> full image, rented-GPU only). Transient/loss masks are
respected the same way the parent loss respects them (masked pixels blacked
out on both sides before the crop).

Kill guard from the plan (PSNR drop > 0.3 dB) is enforced by the experiment
runner comparing eval metrics, not in-model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Type

import torch
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.models.splatfacto_mcmc import SplatfactoMCMCModel, SplatfactoMCMCModelConfig


class PerceptualLossMixin:
    """Adds cfg.lpips_loss_weight * LPIPS_vgg(pred, gt) to the loss dict."""

    def populate_modules(self):
        super().populate_modules()
        self.lpips_vgg_loss = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
        for p in self.lpips_vgg_loss.parameters():
            p.requires_grad_(False)
        self.lpips_vgg_loss.eval()

    def _perceptual_pair(self, outputs, batch):
        """Replicates the parent's gt/pred prep (background composite + mask blackout)."""
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]
        if "mask" in batch:
            mask = self._downscale_if_required(batch["mask"]).to(self.device)
            gt_img = gt_img * mask
            pred_img = pred_img * mask
        return gt_img, pred_img

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        cfg = self.config
        if not self.training or cfg.lpips_loss_weight <= 0 or self.step < cfg.lpips_start_step:
            return loss_dict

        gt_img, pred_img = self._perceptual_pair(outputs, batch)
        gt = gt_img.permute(2, 0, 1).unsqueeze(0)      # (1,3,H,W), [0,1]
        pred = pred_img.permute(2, 0, 1).unsqueeze(0)
        ps = cfg.lpips_patch_size
        if ps and gt.shape[-2] > ps and gt.shape[-1] > ps:
            y = int(torch.randint(0, gt.shape[-2] - ps + 1, ()).item())
            x = int(torch.randint(0, gt.shape[-1] - ps + 1, ()).item())
            gt = gt[..., y:y + ps, x:x + ps]
            pred = pred[..., y:y + ps, x:x + ps]
        loss_dict["lpips_loss"] = cfg.lpips_loss_weight * self.lpips_vgg_loss(
            pred.clamp(0, 1), gt.clamp(0, 1))
        return loss_dict


@dataclass
class SplatfactoPerceptualModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: SplatfactoPerceptualModel)
    lpips_loss_weight: float = 0.1
    """Weight of the LPIPS(VGG) term. Sweep {0.05, 0.1} per the plan."""
    lpips_start_step: int = 0
    """Step to enable the term at (0 = immediately; fine-tune runs start hot)."""
    lpips_patch_size: int = 512
    """Random-crop size for the LPIPS term (0 = full image, needs big VRAM)."""


class SplatfactoPerceptualModel(PerceptualLossMixin, SplatfactoModel):
    config: SplatfactoPerceptualModelConfig


@dataclass
class SplatfactoMCMCPerceptualModelConfig(SplatfactoMCMCModelConfig):
    _target: Type = field(default_factory=lambda: SplatfactoMCMCPerceptualModel)
    lpips_loss_weight: float = 0.1
    lpips_start_step: int = 0
    lpips_patch_size: int = 512


class SplatfactoMCMCPerceptualModel(PerceptualLossMixin, SplatfactoMCMCModel):
    config: SplatfactoMCMCPerceptualModelConfig
