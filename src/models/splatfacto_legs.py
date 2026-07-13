"""exp023: LEGS -- Laplacian-Enhanced Gaussian Splatting with a nonlinear
weighted loss (Guo et al., arXiv:2606.07932). Loss-only, rendering pipeline
unchanged, so implemented as a mixin over SplatfactoModel (same plumbing as the
exp009 perceptual mixin).

Faithful to the paper's equations (no official code exists -- Papers-with-Code
lists none for LEGS or its predecessor EGGS, verified 2026-07-11):

  E_Delta(u,v)  = || Laplacian(F) ||_p                              (Eq. 4)
  Ehat(u,v)     = (E - E_min) / (E_max - E_min + eps)               (Eq. 5)
  Etilde        = f_m(Ehat)                                          (Eq. 6)
  W(u,v)        = 1 + beta * Etilde                                  (Eq. 7)
  L_w           = mean_pixels( W * |U - F|_1 )                       (Eq. 8)
  L_LEGS        = (1 - lambda) * L_w + lambda * L_ssim               (Eq. 9)

F = ground-truth image, U = rendered image. The weight is built from the GT
Laplacian, so it is a fixed per-view map (recomputed each step -- a single
conv2d, negligible vs rasterization).

WHERE THE PAPER IS SILENT (documented choices, exposed as config knobs):
  * Laplacian kernel: the paper says only "the image Laplacian operator". We use
    the standard 4-neighbour discrete Laplacian [[0,1,0],[1,-4,1],[0,1,0]].
  * p in Eq. 4 (RGB channel response magnitude): unspecified -> p=2 (L2 over
    channels), the natural "response vector magnitude".
  * eps in Eq. 5: unspecified -> 1e-6.
  * f_m (C1..C5, Eq. 6 / Fig. 2): the paper gives NO algebraic form for the
    nonlinear maps and reports C3 best (+0.25 dB PSNR over C1). Only C1 (the
    linear/identity map) is fully specified, so C1 is the faithful default. A
    `legs_gamma` power map Etilde = Ehat**gamma is exposed as a concrete
    C3-style concave option (gamma<1) for the sweep; gamma=1 == C1 exactly.
  * beta=10 and lambda=0.2 are the paper's stated bests (lambda matches
    nerfstudio's splatfacto ssim_lambda default, so beta=0 reproduces baseline).

NOTE vs exp022: LEGS deliberately does NOT mean-normalize W (Eq. 8 is a plain
mean of W*|.|), so it also scales up the effective L1 magnitude by mean(W). That
is the paper's design; we keep it. beta=0 -> W==1 -> byte-identical to baseline
(the A/B control).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Type

import torch
import torch.nn.functional as F_nn
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig

# standard 4-neighbour discrete Laplacian
_LAP_KERNEL = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


class LaplacianWeightedLossMixin:
    """Replaces splatfacto's uniform L1 with LEGS' Laplacian-weighted L1 (Eq. 8-9)."""

    def _legs_weight(self, gt_img: torch.Tensor) -> torch.Tensor:
        """gt_img: (H,W,3) in [0,1] -> per-pixel weight (H,W), Eq. 4-7."""
        cfg = self.config
        chw = gt_img.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        k = _LAP_KERNEL.to(gt_img.device, gt_img.dtype).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        lap = F_nn.conv2d(F_nn.pad(chw, (1, 1, 1, 1), mode="reflect"), k, groups=3)  # (1,3,H,W)
        E = torch.linalg.vector_norm(lap, ord=2, dim=1)[0]  # ||.||_2 over channels -> (H,W), Eq. 4
        Emin, Emax = E.min(), E.max()
        Ehat = (E - Emin) / (Emax - Emin + 1e-6)  # Eq. 5, eps=1e-6
        Etilde = Ehat if cfg.legs_gamma == 1.0 else Ehat.clamp_min(0).pow(cfg.legs_gamma)  # Eq. 6
        return 1.0 + cfg.legs_beta * Etilde  # Eq. 7

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        cfg = self.config
        if cfg.legs_beta <= 0:  # control: identical to baseline splatfacto
            return loss_dict

        # replicate the parent's gt/pred prep (composite + optional mask), same as exp009/exp022
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]
        if "mask" in batch:
            mask = self._downscale_if_required(batch["mask"]).to(self.device)
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        W = self._legs_weight(gt_img)  # (H,W)
        Ll1 = (W * torch.abs(gt_img - pred_img).mean(dim=-1)).mean()  # Eq. 8
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None], pred_img.permute(2, 0, 1)[None])
        loss_dict["main_loss"] = (1 - cfg.ssim_lambda) * Ll1 + cfg.ssim_lambda * simloss  # Eq. 9
        return loss_dict


@dataclass
class SplatfactoLEGSModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: SplatfactoLEGSModel)
    legs_beta: float = 10.0
    """Laplacian enhancement strength (paper's best; Eq. 7). 0 = control (== baseline)."""
    legs_gamma: float = 1.0
    """Nonlinear map exponent (Eq. 6). 1.0 = C1 linear (fully specified). <1 = concave C3-style."""


class SplatfactoLEGSModel(LaplacianWeightedLossMixin, SplatfactoModel):
    config: SplatfactoLEGSModelConfig
