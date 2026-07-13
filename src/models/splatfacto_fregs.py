"""exp024: FreGS -- progressive frequency regularization (Zhang et al., FreGS,
CVPR 2024, arXiv:2403.06908). We implement the frequency-regularization LOSS
term (the loss-only, stack-agnostic slice the plan classifies as Tier-1); the
paper also couples this to a frequency-guided densifier, which we do NOT touch
here (default DefaultStrategy densifier is kept). No official or community code
exists (verified 2026-07-11), so this follows the paper's equations directly.

Fourier-space regularization of amplitude and phase discrepancy between the
rendered image U and ground truth F (paper's Sec. 3.3):

  |F(u,v)|  = sqrt(Re^2 + Im^2),   angle F(u,v) = atan2(Im, Re)
  d_a       = (1/sqrt(HW)) * sum | |U_hat| - |F_hat| |            (amplitude)
  d_p       = (1/sqrt(HW)) * sum | angle(U_hat) - angle(F_hat) |  (phase)
  L_f       = w_l (d_la + d_lp)                     for t <= T0
              w_l (d_la + d_lp) + w_h (d_ha + d_hp) for t >  T0   (progressive)
  L_total   = L_3dgs + L_f

Progressive frequency annealing (coarse -> fine): a fixed low-pass band (radius
<= D0) is regularized throughout; a high-pass band (D0 < radius <= D_t) is added
after T0, with the ceiling D_t growing linearly from D0 to the max radius as
t: T0 -> T (paper Eq. for D_t). Radius is normalized to [0,1] on the fftshifted
spectrum (0 = DC at centre, 1 = highest frequency at the corner).

WHERE THE PAPER IS SILENT (documented choices, all exposed as config knobs):
  * FFT is taken per-channel then averaged over channels (paper uses I in R^HxWxC
    but omits the reduction).
  * We use norm="ortho" 2D FFT so amplitudes are O(pixel-scale) and L_f is
    comparable to the L1 term without a huge weight (the raw 1/sqrt(HW) factor of
    an un-normalized FFT still leaves a DC term ~ H*W*mean; ortho is the clean,
    Parseval-consistent normalization).
  * Phase discrepancy is WRAPPED to [-pi, pi] before taking |.| -- mandatory for
    the angle difference to be meaningful/bounded (the paper's atan2 phase is
    2pi-periodic); this also bounds d_p <= pi for stability.
  * Constants the paper does not disclose (w_l, w_h, T0, T, D0): documented
    defaults, all config knobs. freq_weight=0 reproduces baseline exactly.

MEMORY/DENSIFICATION FIX (2026-07-11, after a Kaggle T4 OOM): the paper couples
frequency reg with its OWN frequency-guided densifier. We implement only the
loss and keep nerfstudio's DefaultStrategy densifier -- but the freq loss
inflates the screen-space (means2d) gradients that DefaultStrategy uses to decide
splits, so the Gaussian count explodes and OOMs during a split (this only bit
FreGS; control/legs/mcmc finished fine). Fix: gate the freq loss to start AFTER
densification ends (`freq_start_step`, default = splatfacto `stop_split_at`=15000).
Then 0..15000 is byte-identical to the control (which fits), and after 15000 no
splitting occurs, so the loss cannot drive over-densification -- it acts as a
post-densification spectral refinement term. The progressive anneal is re-windowed
into [freq_start_step, freq_T]. This isolates the loss slice from the densifier we
deliberately did not reimplement; document it as such in the read-out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Type

import torch
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig


class FrequencyRegLossMixin:
    """Adds FreGS' progressive amplitude+phase Fourier loss to the loss dict."""

    def _radius_masks(self, H: int, W: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """(low_band, radius) on the fftshifted spectrum; radius in [0,1]. Cached per (H,W)."""
        cache = getattr(self, "_fregs_cache", None)
        if cache is not None and cache[0] == (H, W):
            return cache[1], cache[2]
        fy = torch.fft.fftshift(torch.fft.fftfreq(H, device=device))  # [-0.5,0.5)
        fx = torch.fft.fftshift(torch.fft.fftfreq(W, device=device))
        ry, rx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(ry ** 2 + rx ** 2)
        radius = radius / radius.max()  # normalize to [0,1]
        low_band = radius <= self.config.freq_D0
        self._fregs_cache = ((H, W), low_band, radius)
        return low_band, radius

    @staticmethod
    def _amp_phase(img_hw3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(H,W,3)[0,1] -> per-channel amplitude & phase, fftshifted. -> (3,H,W) each."""
        x = img_hw3.permute(2, 0, 1)  # (3,H,W)
        Fc = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
        return Fc.abs(), torch.atan2(Fc.imag, Fc.real)

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        cfg = self.config
        # Gate off during (and before) densification so the freq-loss gradient cannot
        # inflate DefaultStrategy's split decisions -> no Gaussian-count OOM. See module docstring.
        if not self.training or cfg.freq_weight <= 0 or self.step < cfg.freq_start_step:
            return loss_dict

        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]
        if "mask" in batch:
            mask = self._downscale_if_required(batch["mask"]).to(self.device)
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        H, W = gt_img.shape[:2]
        low_band, radius = self._radius_masks(H, W, gt_img.device)
        amp_u, ph_u = self._amp_phase(pred_img)
        amp_f, ph_f = self._amp_phase(gt_img)

        d_amp = (amp_u - amp_f).abs().mean(dim=0)              # (H,W), channel-averaged
        dphi = ph_u - ph_f
        d_phase = torch.atan2(torch.sin(dphi), torch.cos(dphi)).abs().mean(dim=0)  # wrapped, (H,W)

        def band_terms(band):
            n = band.sum().clamp_min(1)
            return d_amp[band].sum() / n, d_phase[band].sum() / n  # mean over the band

        d_la, d_lp = band_terms(low_band)
        L_f = cfg.freq_w_low * (d_la + d_lp)

        # progressive high band: ceiling grows D0 -> 1 over (T0, T]
        if self.step > cfg.freq_T0:
            frac = min(1.0, (self.step - cfg.freq_T0) / max(1, cfg.freq_T - cfg.freq_T0))
            D_t = cfg.freq_D0 + frac * (1.0 - cfg.freq_D0)
            high_band = (radius > cfg.freq_D0) & (radius <= D_t)
            if high_band.any():
                d_ha, d_hp = band_terms(high_band)
                L_f = L_f + cfg.freq_w_high * (d_ha + d_hp)

        loss_dict["freq_loss"] = cfg.freq_weight * L_f
        return loss_dict


@dataclass
class SplatfactoFreGSModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: SplatfactoFreGSModel)
    freq_weight: float = 0.05
    """Global scale on L_f (0 = control == baseline). Swept in the notebook."""
    freq_w_low: float = 1.0
    """w_l: weight of the low-frequency band term."""
    freq_w_high: float = 1.0
    """w_h: weight of the progressively-revealed high-frequency band term."""
    freq_D0: float = 0.25
    """Normalized radius of the fixed low-pass band (inner 25% of the spectrum)."""
    freq_start_step: int = 15000
    """Step to activate the whole freq loss. Default = splatfacto stop_split_at, so the loss
    never overlaps densification (prevents the Gaussian-count OOM). Set lower only if the
    densifier is also constrained (e.g. MCMC cap) or for a short smoke test."""
    freq_T0: int = 15000
    """Step to start adding the high-frequency band (coarse->fine anneal start)."""
    freq_T: int = 27000
    """Step by which the high-band ceiling reaches the full spectrum (anneal end)."""


class SplatfactoFreGSModel(FrequencyRegLossMixin, SplatfactoModel):
    config: SplatfactoFreGSModelConfig
