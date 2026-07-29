"""Apply forward SIMPLE_RADIAL distortion to pinhole renders so they match the
raw distorted DJI test GT. Reproduces the out_k block of
bts-nvs/Analysis/04_x3_dibr_pilot.py (lines ~491-515) standalone.

make_undistorted_scene keeps K UNCHANGED (f,cx,cy identical to raw; only pixels
resampled), so the pinhole render shares the raw camera's intrinsics and the
only correction needed is the radial resample below.

For each distorted output pixel (u,v): normalized xd=(u-cx)/f; solve
xu = xd/(1+k*r_u^2) (Newton-style fixed point, 5 iters, matches Warper); then
sample the pinhole render at (xu*f+cx, yu*f+cy). Positive k -> samples inward,
never out of bounds, so no canvas margin.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np


def build_maps(W, H, f, cx, cy, k):
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    xd = (u - cx) / f
    yd = (v - cy) / f
    xu, yu = xd.copy(), yd.copy()
    for _ in range(5):
        r2 = xu * xu + yu * yu
        xu = xd / (1 + k * r2)
        yu = yd / (1 + k * r2)
    us = (xu * f + cx).astype(np.float32)
    vs = (yu * f + cy).astype(np.float32)
    return us, vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="dir of pinhole PNG renders")
    ap.add_argument("--dst", required=True, type=Path, help="output dir (distorted)")
    ap.add_argument("--f", type=float, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--H", type=int, required=True)
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    us, vs = build_maps(args.W, args.H, args.f, args.cx, args.cy, args.k)
    pngs = sorted(args.src.glob("*.png"))
    assert pngs, f"no PNGs in {args.src}"
    for p in pngs:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        assert img.shape[1] == args.W and img.shape[0] == args.H, \
            f"{p.name}: {img.shape[1]}x{img.shape[0]} != {args.W}x{args.H}"
        out = cv2.remap(img.astype(np.float32), us, vs, cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE)
        out = np.clip(out, 0, 255).astype(np.uint8)
        cv2.imwrite(str(args.dst / p.name), out)
    print(f"remapped {len(pngs)} -> {args.dst}  (f={args.f} cx={args.cx} cy={args.cy} k={args.k})")


if __name__ == "__main__":
    main()
