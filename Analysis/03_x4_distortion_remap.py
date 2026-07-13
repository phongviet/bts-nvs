"""Analysis 03 / Experiment X4: are the test GT images RAW (radially distorted)?

Hypothesis: scene cameras are COLMAP SIMPLE_RADIAL (k ~ 0.008-0.010). nerfstudio
undistorts train images before training and renders PINHOLE at test poses. If
the test GT images are raw DJI frames (distorted), every submitted render is
misaligned by up to ~4-5 px at frame corners — which is exactly where the
exp020 weakness deep-dive measured 2.5-2.9x error. The fix would be free points
on all 13 scenes: remap renders into distorted geometry before submission.

Method: take the existing locked-config test renders (undistorted pinhole),
synthesize their distorted counterparts (for each distorted-image pixel,
invert SIMPLE_RADIAL by fixed-point iteration, sample the pinhole render),
score BOTH against GT with the confirmed grader metric (vgg + psnr_max 50),
and also compare corner-tile MSE specifically.

Run: conda run -n airace python Analysis/03_x4_distortion_remap.py [scene]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "phase1" / "public_set"
OUT = Path(__file__).resolve().parent / "X4_distortion_remap"
sys.path.insert(0, str(REPO))

from src.metrics import compute_metrics  # noqa: E402
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary  # noqa: E402

RENDER_DIRS = {
    "hcm0034": REPO / "runs/phase1/exp004_backend_ablation/antialiased/renders_test",
    "hcm0031": REPO / "runs/phase1/exp004_backend_ablation/hcm0031_antialiased/renders_test",
    "HCM0181": REPO / "runs/phase1/exp004_backend_ablation/HCM0181_antialiased/renders_test",
    "HCM0193": REPO / "runs/phase1/exp004_backend_ablation/HCM0193_antialiased/renders_test",
    "HCM0204": REPO / "runs/phase1/exp004_hcm0204_fill/HCM0204/antialiased/renders_test",
}


def distort_remap(img: np.ndarray, fx, fy, cx, cy, k: float) -> np.ndarray:
    """Given a pinhole (undistorted) render, produce the SIMPLE_RADIAL image."""
    H, W = img.shape[:2]
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    xd = (u - cx) / fx  # normalized coords of the DISTORTED (output) image
    yd = (v - cy) / fy
    # invert x_d = x_u * (1 + k r_u^2) by fixed point: x_u = x_d / (1 + k r_u^2)
    xu, yu = xd.copy(), yd.copy()
    for _ in range(5):
        r2 = xu * xu + yu * yu
        xu = xd / (1 + k * r2)
        yu = yd / (1 + k * r2)
    us = xu * fx + cx
    vs = yu * fy + cy
    # bilinear sample
    u0 = np.clip(np.floor(us).astype(int), 0, W - 2)
    v0 = np.clip(np.floor(vs).astype(int), 0, H - 2)
    du = np.clip(us - u0, 0, 1)[..., None]
    dv = np.clip(vs - v0, 0, 1)[..., None]
    im = img.astype(np.float64)
    out = (im[v0, u0] * (1 - du) * (1 - dv) + im[v0, u0 + 1] * du * (1 - dv)
           + im[v0 + 1, u0] * (1 - du) * dv + im[v0 + 1, u0 + 1] * du * dv)
    return out.astype(np.uint8)


def corner_mse(render_dir: Path, gt_dir: Path, frac=0.15) -> float:
    """mean MSE over the two TOP corner tiles (the measured weak spot)."""
    vals = []
    for rf in sorted(render_dir.glob("*.JPG")):
        gt = np.asarray(Image.open(gt_dir / rf.name).convert("RGB"), dtype=np.float32) / 255
        pr = np.asarray(Image.open(rf).convert("RGB"), dtype=np.float32) / 255
        h, w = gt.shape[:2]
        th, tw = int(h * frac), int(w * frac)
        for sl in [(slice(0, th), slice(0, tw)), (slice(0, th), slice(w - tw, w))]:
            vals.append(float(((gt[sl] - pr[sl]) ** 2).mean()))
    return float(np.mean(vals))


def main():
    scenes = sys.argv[1:] or ["hcm0034"]
    results = {}
    for scene in scenes:
        rdir = RENDER_DIRS[scene]
        gt_dir = RAW / scene / "test" / "images"
        cams = read_cameras_binary(RAW / scene / "train" / "sparse" / "0" / "cameras.bin")
        cam = list(cams.values())[0]
        assert cam.model == "SIMPLE_RADIAL", cam.model
        f, cx, cy, k = cam.params
        print(f"{scene}: SIMPLE_RADIAL f={f:.3f} cx={cx:.1f} cy={cy:.1f} k={k:.5f} "
              f"({len(cams)} camera entries)")

        ddir = OUT / scene
        ddir.mkdir(parents=True, exist_ok=True)
        for rf in sorted(rdir.glob("*.JPG")):
            img = np.asarray(Image.open(rf).convert("RGB"))
            Image.fromarray(distort_remap(img, f, f, cx, cy, k)).save(ddir / rf.name, quality=98)

        base = compute_metrics(rdir, gt_dir, "vgg", 50.0)["mean"]
        dist = compute_metrics(ddir, gt_dir, "vgg", 50.0)["mean"]
        cm_base, cm_dist = corner_mse(rdir, gt_dir), corner_mse(ddir, gt_dir)
        results[scene] = {"pinhole": base, "distorted": dist,
                          "corner_mse_pinhole": cm_base, "corner_mse_distorted": cm_dist}
        print(f"  pinhole : PSNR={base['psnr']:.3f} SSIM={base['ssim']:.4f} "
              f"LPIPS={base['lpips']:.4f} Score={base['score']:.4f} cornerMSE={cm_base:.5f}")
        print(f"  distorted: PSNR={dist['psnr']:.3f} SSIM={dist['ssim']:.4f} "
              f"LPIPS={dist['lpips']:.4f} Score={dist['score']:.4f} cornerMSE={cm_dist:.5f}")
        print(f"  Δscore={dist['score']-base['score']:+.4f}  ΔcornerMSE={cm_dist-cm_base:+.5f}")

    (OUT / "summary.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
