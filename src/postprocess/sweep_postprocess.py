"""exp011 (A.7): post-process + encoding sweep on EXISTING renders -- zero
training cost. For each (op x encoder) variant: process the scene's
renders_test into a temp dir, score against public GT, append a CSV row.

Also runs the color/gamma round-trip check first (--roundtrip): pass a GT
image itself through each encoder and report its PSNR -- if 'identity+jpeg95'
on GT scores < ~50 dB something in the IO path (gamma/ICC) is broken, and
op deltas can't be trusted.

Usage:
  python src/postprocess/sweep_postprocess.py \
      --renders runs/phase1/exp002_dense_colmap_init/hcm0034/renders_test \
      --gt data/raw/phase1/public_set/hcm0034/test/images \
      --scene hcm0034 --out results/week3_postproc_ablation.csv \
      [--ops identity unsharp_r1_p50 ...] [--encoders jpeg75 jpeg95 png]

Keep ONE global winner (op, encoder) across scenes; per-scene cherry-picking
here is noise-chasing (deltas are ~1e-3).
"""
import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.metrics import compute_metrics, load_rgb  # noqa: E402
from src.postprocess.ops import ENCODERS, OPS, process_dir, process_image  # noqa: E402

CSV_FIELDS = ["scene", "exp", "variant", "op", "encoder",
              "psnr", "ssim", "lpips", "score", "n_images"]


def roundtrip_check(gt_dir: Path, encoders: list[str]):
    """Encode a GT image with each encoder and PSNR it against itself."""
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    gt_path = sorted(gt_dir.iterdir())[0]
    gt = load_rgb(gt_path)
    print(f"round-trip check on {gt_path.name}:")
    with tempfile.TemporaryDirectory() as td:
        for enc in encoders:
            out = Path(td) / gt_path.name
            process_image(gt_path, out, "identity", enc)
            back = load_rgb(out)
            psnr = float(sk_psnr(gt, back, data_range=1.0)) if not np.array_equal(gt, back) else float("inf")
            print(f"  identity+{enc:12s}: {psnr:.2f} dB")
            if psnr < 45:
                print(f"  !! {enc}: round-trip below 45 dB -- check gamma/ICC handling before trusting deltas")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--exp", default="exp011")
    ap.add_argument("--out", type=Path, default=Path("results/week3_postproc_ablation.csv"))
    ap.add_argument("--ops", nargs="*", default=list(OPS.keys()))
    ap.add_argument("--encoders", nargs="*", default=["jpeg75", "jpeg90", "jpeg95", "jpeg98",
                                                      "jpeg95_444", "png"])
    ap.add_argument("--lpips-net", default="vgg")  # LB backbone, confirmed 2026-07-09
    ap.add_argument("--psnr-max", type=float, default=50.0)
    ap.add_argument("--roundtrip", action="store_true")
    args = ap.parse_args()

    if args.roundtrip:
        roundtrip_check(args.gt, args.encoders)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        for op in args.ops:
            for enc in args.encoders:
                with tempfile.TemporaryDirectory() as td:
                    n = process_dir(args.renders, Path(td), op, enc)
                    m = compute_metrics(Path(td), args.gt, args.lpips_net, args.psnr_max)["mean"]
                row = {"scene": args.scene, "exp": args.exp, "variant": f"{op}+{enc}",
                       "op": op, "encoder": enc, "n_images": n,
                       "psnr": round(m["psnr"], 4), "ssim": round(m["ssim"], 5),
                       "lpips": round(m["lpips"], 5), "score": round(m["score"], 5)}
                w.writerow(row)
                f.flush()
                print(f"{args.scene} {row['variant']:24s} score={row['score']:.5f} "
                      f"psnr={row['psnr']:.3f} ssim={row['ssim']:.4f} lpips={row['lpips']:.4f}")


if __name__ == "__main__":
    main()
