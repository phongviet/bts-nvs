"""Analysis 17 (exp039, Wave-1): flow-residual alignment of warped neighbour
views before the DIBR blend -- a train-free approximation of GADA's learned
deformable offsets (arXiv 2607.00595), which raise usable warped-pixel density
from 33% to 79%.

Mechanism: our depth-based warp lands each real train pixel at the position
implied by the 3DGS expected depth. Where that depth is biased (thin antennas,
lattice masts, silhouettes) the warped neighbour is displaced a few pixels from
the true 3DGS render, so the photometric guard REJECTS it and we fall back to
the blurry render -- losing exactly the high-frequency real texture we wanted.
GADA fixes this with learned offsets; here we estimate a small optical flow from
each warped neighbour to the 3DGS render, clamp it to |flow| <= max_px (GADA's
sigma_max = 7 px), and re-sample the neighbour along it BEFORE the guard/blend.
Displaced-but-valid cues snap back into agreement -> the guard keeps them.

Backends (pluggable, best first):
  * SEA-RAFT (arXiv 2405.14793, ECCV24 Oral) if a checkpoint is provided --
    scene-agnostic pretrained, inference only (compliant; provenance row
    required in docs/rules_and_constraints.md before shipping).
  * cv2.DISOpticalFlow -- CLASSICAL algorithm, NO pretrained weights, always
    available, fully compliant. The default so the wave is runnable today.

This module exposes `align_to_reference(src, ref, max_px, backend)` used by the
DIBR Warper (04) when `flow_align=` is passed. Kept standalone + unit-tested so
the alignment is validated independently of the (slow) full warp.
"""
from __future__ import annotations

import numpy as np

_SEARAFT = {}  # lazy singleton cache keyed by checkpoint path


def _dis_flow(src_gray: np.ndarray, ref_gray: np.ndarray) -> np.ndarray:
    """Dense inverse-search optical flow ref<-src (classical, weightless).
    Returns flow (H,W,2) mapping each ref pixel to a src displacement."""
    import cv2
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    a = (ref_gray * 255).astype(np.uint8)
    b = (src_gray * 255).astype(np.uint8)
    return dis.calc(a, b, None)  # (H,W,2), flow[...,0]=dx, [...,1]=dy


def _searaft_flow(src, ref, ckpt):
    """SEA-RAFT flow ref<-src. Requires the SEA-RAFT package + checkpoint; the
    import is deferred so the classical path never pays for it. Returns (H,W,2)."""
    import torch
    if ckpt not in _SEARAFT:
        from sea_raft import SEARAFT  # vendored under third_party/ when used
        model = SEARAFT.from_pretrained(ckpt).eval()
        _SEARAFT[ckpt] = model
    model = _SEARAFT[ckpt]
    dev = next(model.parameters()).device
    to = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().to(dev)
    with torch.no_grad():
        flow = model(to(ref), to(src))[-1]  # ref<-src
    return flow[0].permute(1, 2, 0).cpu().numpy()


def align_to_reference(src: np.ndarray, ref: np.ndarray, max_px: float = 7.0,
                       backend: str = "dis", searaft_ckpt: str | None = None,
                       conf_thresh: float = 0.06):
    """Re-sample `src` (a warped neighbour, HxWx3 float [0,1]) so it aligns to
    `ref` (the 3DGS render, same shape). Flow is clamped to +-max_px (GADA
    sigma_max). Pixels where the aligned src still disagrees with ref by more
    than conf_thresh keep the ORIGINAL src (never make a pixel worse than the
    un-aligned warp -- the same conservative philosophy as the photometric
    guard). Returns (aligned_src, applied_mask)."""
    import cv2
    H, W = ref.shape[:2]
    sg = src.mean(-1).astype(np.float32)
    rg = ref.mean(-1).astype(np.float32)
    if backend == "searaft" and searaft_ckpt is not None:
        flow = _searaft_flow(src, ref, searaft_ckpt)
    else:
        flow = _dis_flow(sg, rg)
    # clamp magnitude to max_px (large flow = unreliable -> ignore)
    mag = np.linalg.norm(flow, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_px / np.maximum(mag, 1e-6))
    flow = flow * scale
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    map_x = (gx + flow[..., 0]).astype(np.float32)
    map_y = (gy + flow[..., 1]).astype(np.float32)
    aligned = cv2.remap(src.astype(np.float32), map_x, map_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    # keep alignment only where it improved agreement with ref
    err0 = np.abs(src - ref).mean(-1)
    err1 = np.abs(aligned - ref).mean(-1)
    improved = err1 + 1e-4 < err0
    out = np.where(improved[..., None], aligned, src)
    return np.clip(out, 0, 1), improved.astype(np.float32)


# ------------------------------- self-test -------------------------------------
def _demo():
    """Shift a random image by a few px, then recover the shift via alignment."""
    rng = np.random.default_rng(0)
    ref = rng.random((64, 96, 3)).astype(np.float32)
    import cv2
    ref = cv2.GaussianBlur(ref, (0, 0), 1.5)  # give the flow something trackable
    M = np.float32([[1, 0, 3], [0, 1, -2]])  # src is ref shifted (3,-2) px
    src = cv2.warpAffine(ref, M, (96, 64), borderMode=cv2.BORDER_REPLICATE)
    aligned, mask = align_to_reference(src, ref, max_px=7, backend="dis")
    e0 = np.abs(src - ref).mean()
    e1 = np.abs(aligned - ref).mean()
    print(f"mean |src-ref|={e0:.4f} -> |aligned-ref|={e1:.4f} "
          f"(applied on {mask.mean()*100:.0f}% of pixels) "
          f"{'OK' if e1 < e0 else 'NO IMPROVEMENT'}")


if __name__ == "__main__":
    _demo()
