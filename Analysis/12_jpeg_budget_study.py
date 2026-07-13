"""Analysis 12: JPEG bit-budget study (E1 of the top-1 ladder).

The 350 MB submission cap forced exp033 down to q95 (uniform), costing a
measured -0.003 Score. Two inefficiencies to reclaim:
  1. double compression: apply_test saves q98 JPEGs, then 11_build re-encodes
     them at q95 (two generations of loss);
  2. uniform quality: the right knob is "highest q whose PRIVATE total fits
     ~340 MB", possibly with subsampling / optimize=True (free ~5-8% size).

This script measures, on public scenes with GT, Score(encode-config) using the
existing q98 renders_refined as source (matching what a re-encode of current
outputs gives), and projects the private zip size for each config from the
actual 434 private refined images. Output: the best config <= budget.

Run: conda run -n airace python Analysis/12_jpeg_budget_study.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.metrics import compute_metrics  # noqa: E402

REFINER = REPO / "Analysis/X5_refiner"
OUT = REPO / "Analysis/X6_jpeg_budget"
PUBLIC = ["hcm0031", "hcm0034", "HCM0181", "HCM0193", "HCM0204"]
PRIVATE = ["HCM0249", "HCM0254", "HCM0276", "HCM1439", "HNI0131", "HNI0265", "HNI0366", "HNI0437"]
RAW_PUB = REPO / "data/raw/phase1/public_set"

# (name, quality, subsampling, optimize)  subsampling: 0=4:4:4, 2=4:2:0
CONFIGS = [
    ("q95_sub2", 95, 2, False),          # exp033 as-shipped (PIL default sub for q95)
    ("q95_sub2_opt", 95, 2, True),
    ("q96_sub2_opt", 96, 2, True),
    ("q97_sub2_opt", 97, 2, True),
    ("q98_sub2_opt", 98, 2, True),
    ("q93_sub0_opt", 93, 0, True),
    ("q95_sub0_opt", 95, 0, True),
    ("q96_sub0_opt", 96, 0, True),
    ("q97_sub0_opt", 97, 0, True),
]
BUDGET_MB = 340  # keep 10 MB slack under the 350 MB rule


def encode_size(img: Image.Image, q, sub, opt) -> int:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q, subsampling=sub, optimize=opt)
    return buf.getbuffer().nbytes


def main():
    OUT.mkdir(exist_ok=True)
    # ---- private size projection per config (all 434 images, exact) ----
    sizes = {name: 0 for name, *_ in CONFIGS}
    n_priv = 0
    for s in PRIVATE:
        for f in sorted((REFINER / s / "renders_refined").glob("*.JPG")):
            im = Image.open(f).convert("RGB")
            n_priv += 1
            for name, q, sub, opt in CONFIGS:
                sizes[name] += encode_size(im, q, sub, opt)
    print(f"private images: {n_priv}")
    for name, *_ in CONFIGS:
        mb = sizes[name] / 1e6
        flag = "OK " if mb <= BUDGET_MB else "OVER"
        print(f"  {name:14s} projected private zip ~{mb:6.1f} MB  [{flag}]")

    # ---- Score per config on 2 public scenes (fast proxy: worst + best LPIPS) ----
    results = {}
    for scene in ["hcm0034", "HCM0193"]:
        gt = RAW_PUB / scene / "test/images"
        src = REFINER / scene / "renders_refined"
        for name, q, sub, opt in CONFIGS:
            d = OUT / f"{scene}_{name}"
            d.mkdir(exist_ok=True)
            for f in sorted(src.glob("*.JPG")):
                Image.open(f).convert("RGB").save(d / f.name, "JPEG",
                                                  quality=q, subsampling=sub, optimize=opt)
            m = compute_metrics(d, gt, "vgg", 50.0)["mean"]
            results[f"{scene}/{name}"] = m
            print(f"{scene} {name:14s} Score={m['score']:.5f} PSNR={m['psnr']:.3f} "
                  f"SSIM={m['ssim']:.4f} LPIPS={m['lpips']:.4f}")

    (OUT / "summary.json").write_text(json.dumps(
        {"private_bytes": sizes, "n_private_imgs": n_priv, "public": results}, indent=2))
    print("done -> Analysis/X6_jpeg_budget/summary.json")


if __name__ == "__main__":
    main()
