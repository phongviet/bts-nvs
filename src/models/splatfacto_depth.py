"""Splatfacto + scale-and-shift-invariant monocular-depth loss (backbone-side plan §2.2).

Motivation. The SoccerNet 2026 NVS challenge winner (DENSER) used scale-and-shift-invariant
Depth-Anything-V2 supervision "to regularize geometry in textureless regions". chair's recorded
failure is exactly that: MVS ran out of confident geometry (66 MB fused.ply, smallest of the 7
scenes; hold-out Score 0.6506, the worst). splatfacto has NO geometry supervision at all -- it
only *outputs* depth -- so this is a genuinely untouched axis, not a re-tune.

Why scale-and-shift-INVARIANT. Monocular depth is relative: it has an unknown affine transform
per image (DRGS / DRGSplat / Cascade-Pearson all agree it is unusable as absolute geometry but
reliable as relative structure). We therefore solve a per-step least-squares fit for (a, b) in
    a * mono_disp + b  ~=  rendered_disp
and penalise the residual. Any affine transform of the stored map is absorbed -- which is also
why the exporter is free to min-max normalise into 16-bit PNG.

Why DISPARITY space. Rendered depth is a z-buffer; mono-depth-v2 predicts relative INVERSE
depth. Comparing in disparity space keeps the near field (where chair's detail lives) weighted
sensibly and avoids the 1/z blow-up at the background.

Depth maps are loaded through nerfstudio's own `--depths-path` plumbing, so
`metadata["depth_filenames"]` is INDEX-ALIGNED with the images by construction. We never guess
the dataparser's ordering.

Instrumentation is not optional: exp009 was voided by a silent no-op that looked like a
measured null result. `depth_trace_every` prints the live term and the first step asserts it is
attached to the graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig


def _ssi_align(src: torch.Tensor, tgt: torch.Tensor):
    """Least-squares (a, b) minimising ||a*src + b - tgt||^2 over the given 1-D samples."""
    ones = torch.ones_like(src)
    A = torch.stack([src, ones], dim=1)                      # (N,2)
    ata = A.T @ A                                            # (2,2)
    # Ill-conditioned when src is ~constant (a flat depth map); fall back to shift-only.
    if float(torch.det(ata).abs()) < 1e-8:
        return torch.ones((), device=src.device), (tgt.mean() - src.mean())
    sol = torch.linalg.solve(ata, A.T @ tgt)
    return sol[0], sol[1]


class DepthLossMixin:
    """Adds cfg.depth_loss_weight * L1(ssi_align(mono_disp), rendered_disp)."""

    def populate_modules(self):
        super().populate_modules()
        self._depth_maps: Optional[list] = None
        # splatfacto skips the depth channel during training unless asked -- without this the
        # loss silently has nothing to consume.
        self.config.output_depth_during_training = True

    def _load_depths(self):
        from PIL import Image
        import numpy as np
        meta = getattr(self, "kwargs", {}).get("metadata", {}) or {}
        files = meta.get("depth_filenames")
        assert files, (
            "no depth_filenames in dataparser metadata -- pass `--depths-path <dir>` to the "
            "colmap dataparser so nerfstudio builds an INDEX-ALIGNED depth list.")
        maps = []
        for f in files:
            d = torch.from_numpy(np.asarray(Image.open(f)).astype("float32"))
            maps.append(d / 65535.0)
        self._depth_maps = maps
        print(f"[depth] loaded {len(maps)} mono-depth maps, first {tuple(maps[0].shape)}", flush=True)

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        cfg = self.config
        if not self.training or cfg.depth_loss_weight <= 0 or self.step < cfg.depth_start_step:
            return loss_dict
        if self._depth_maps is None:
            self._load_depths()

        rendered = outputs.get("depth")
        assert rendered is not None, "no depth in outputs -- output_depth_during_training is off"
        idx = int(batch["image_idx"])
        mono = self._depth_maps[idx].to(self.device)                      # (H,W) relative inverse
        rd = rendered.squeeze(-1)                                          # (H,W) z
        if mono.shape != rd.shape:
            mono = torch.nn.functional.interpolate(
                mono[None, None], size=rd.shape, mode="bilinear", align_corners=False)[0, 0]

        # Only supervise pixels the renderer actually covered; empty space has no opinion.
        acc = outputs.get("accumulation")
        valid = (acc.squeeze(-1) > cfg.depth_acc_thresh) if acc is not None else torch.ones_like(rd, dtype=torch.bool)
        valid = valid & torch.isfinite(rd) & (rd > 1e-6)
        if int(valid.sum()) < 1024:
            return loss_dict

        rd_disp = 1.0 / rd.clamp_min(1e-6)
        s, t = _ssi_align(mono[valid].detach(), rd_disp[valid].detach())
        aligned = (s * mono + t)[valid]
        term = cfg.depth_loss_weight * torch.abs(aligned.detach() - rd_disp[valid]).mean()

        if cfg.depth_trace_every and self.step % cfg.depth_trace_every == 0:
            print(f"[depth-trace] step {self.step} term {float(term):.6f} "
                  f"valid {int(valid.sum())}/{valid.numel()} a={float(s):.4f} b={float(t):.4f} "
                  f"requires_grad={term.requires_grad}", flush=True)
        if self.step == cfg.depth_start_step:
            assert term.requires_grad, (
                "depth term is DETACHED -- it cannot affect the parameters (the exp009 no-op).")
            assert torch.isfinite(term), f"depth term non-finite: {float(term)}"
        loss_dict["depth_loss"] = term
        return loss_dict


@dataclass
class SplatfactoDepthModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: SplatfactoDepthModel)
    depth_loss_weight: float = 0.5
    """Weight of the scale-shift-invariant depth term."""
    depth_start_step: int = 0
    """Step to enable the term at."""
    depth_acc_thresh: float = 0.5
    """Only supervise pixels whose accumulated alpha exceeds this."""
    depth_trace_every: int = 0
    """Print the live term every N steps (0 = off). REQUIRED for any real run."""


class SplatfactoDepthModel(DepthLossMixin, SplatfactoModel):
    config: SplatfactoDepthModelConfig
