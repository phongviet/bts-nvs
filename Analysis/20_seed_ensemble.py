"""Analysis 20 (exp036 / reserve R2): apply-time seed-ensemble average.

Averages the rendered outputs of N refiner seed members pixel-wise (all members
share the same DIBR base, so averaging outputs == averaging residuals) and
scores the ensemble against test GT with the grader metric.

Run: conda run -n airace python Analysis/20_seed_ensemble.py --scene hcm0034 \
    --members renders_refined_v2 renders_refined_v2s1 --out renders_refined_v2ens
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.metrics import compute_metrics  # noqa: E402

OUT = REPO / "Analysis/X5_refiner"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="hcm0034")
    ap.add_argument("--members", nargs="+", required=True,
                    help="render dir names under Analysis/X5_refiner/<scene>/")
    ap.add_argument("--out", required=True, help="output dir name (same root)")
    args = ap.parse_args()

    root = OUT / args.scene
    dirs = [root / m for m in args.members]
    for d in dirs:
        assert d.is_dir(), f"missing member dir: {d}"
    # members may hold both .JPG and lossless .png twins; ensemble over the
    # names ALL members share (the .JPG set in practice)
    names = sorted(set.intersection(*({p.name for p in d.iterdir()} for d in dirs)))
    assert names, "no common filenames across members"
    outdir = root / args.out
    outdir.mkdir(exist_ok=True)
    for n in names:
        acc = None
        for d in dirs:
            a = np.asarray(Image.open(d / n).convert("RGB"), dtype=np.float64)
            acc = a if acc is None else acc + a
        Image.fromarray((acc / len(dirs)).round().astype(np.uint8)).save(
            outdir / n, quality=98)

    gt = REPO / "data/raw/phase1/public_set" / args.scene / "test/images"
    if gt.exists():
        m = compute_metrics(outdir, gt, "vgg", 50.0)["mean"]
        (root / f"metrics_{args.out}.json").write_text(json.dumps(m, indent=2))
        print(f"{args.scene} ENSEMBLE({len(dirs)} members): PSNR={m['psnr']:.3f} "
              f"SSIM={m['ssim']:.4f} LPIPS={m['lpips']:.4f} Score={m['score']:.4f}")
    else:
        print(f"{args.scene}: ensemble written to {outdir} (no GT)")


if __name__ == "__main__":
    main()
