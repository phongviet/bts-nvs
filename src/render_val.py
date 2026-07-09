"""Render a trained checkpoint at the scene's val-split poses and score
against the real photos -- the model-selection signal for private scenes.

Usage:
  python src/render_val.py --config <run>/config.yml \
      --scene-dir data/raw/phase1/private_set1/HCM0249 \
      --processed-root data/processed/phase1 \
      --out <run_dir>/renders_val_split \
      [--metrics-out <run_dir>/metrics_val_split.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.render_utils import load_colmap_poses, render_pose_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--scene-dir", required=True, type=Path)
    ap.add_argument("--processed-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--metrics-out", type=Path, default=None)
    ap.add_argument("--lpips-net", default="vgg")  # LB backbone, confirmed 2026-07-09
    ap.add_argument("--psnr-max", type=float, default=50.0)
    args = ap.parse_args()

    scene = args.scene_dir.name
    scene_root = args.processed_root / scene
    val_ids_file = scene_root / "splits" / "val_ids.txt"
    if not val_ids_file.exists():
        raise SystemExit(f"{val_ids_file} missing -- run make_val_split.py (match-test) first.")
    val_ids = set(val_ids_file.read_text().split())

    rows = load_colmap_poses(scene_root / "colmap_train_only", only_names=val_ids)
    n = render_pose_rows(args.config, rows, args.out)
    print(f"{scene}: rendered {n} val images (of {len(rows)}) -> {args.out}")

    from src.metrics import compute_metrics
    result = compute_metrics(args.out, args.scene_dir / "train" / "images",
                             args.lpips_net, args.psnr_max)
    m = result["mean"]
    out_json = args.metrics_out or (args.out.parent / "metrics_val_split.json")
    out_json.write_text(json.dumps(result, indent=2))
    print(f"VAL {scene}: PSNR={m['psnr']:.3f} SSIM={m['ssim']:.4f} "
          f"LPIPS={m['lpips']:.4f} Score={m['score']:.4f} -> {out_json}")


if __name__ == "__main__":
    main()
