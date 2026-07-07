"""Compute per-train-image sampling weights for exp013 (test-pose-weighted
training). weight_i = 1 + boost * (test poses covered by train view i) /
(max over train views), where "covered" = within dist_frac and angle
thresholds (same definition as test_pose_coverage.py's n_near, transposed).

Writes <staging-dir>/train_weights.json ({image_filename: weight}) which
WeightedFullImageDatamanager picks up automatically.

Usage:
  python src/data_prep/compute_train_weights.py \
      --sparse-dir data/processed/phase1/hcm0034/colmap_train_only \
      --test-poses data/raw/phase1/public_set/hcm0034/test/test_poses.csv \
      --out data/processed/phase1/hcm0034/train_staging_dense/train_weights.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.pose_utils import (  # noqa: E402
    angular_gap_deg, load_poses_from_colmap, load_poses_from_csv, scene_extent,
)


def compute_weights(train_poses, test_poses, boost: float,
                    dist_frac_thresh: float, angle_thresh_deg: float) -> dict[str, float]:
    extent = scene_extent(train_poses.centers)
    d = np.linalg.norm(train_poses.centers[:, None, :] - test_poses.centers[None, :, :], axis=2)
    ang = angular_gap_deg(train_poses.view_dirs, test_poses.view_dirs)
    covered = ((d / extent < dist_frac_thresh) & (ang < angle_thresh_deg)).sum(axis=1)  # per train
    if covered.max() == 0:
        return {n: 1.0 for n in train_poses.names}
    w = 1.0 + boost * covered / covered.max()
    return {n: round(float(x), 4) for n, x in zip(train_poses.names, w)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse-dir", required=True, type=Path, help="colmap_train_only dir")
    ap.add_argument("--test-poses", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--boost", type=float, default=1.0,
                    help="max extra weight (1.0 -> best-covering image sampled 2x baseline)")
    ap.add_argument("--dist-frac", type=float, default=0.05)
    ap.add_argument("--angle-deg", type=float, default=20.0)
    args = ap.parse_args()

    train = load_poses_from_colmap(args.sparse_dir)
    test = load_poses_from_csv(args.test_poses)
    weights = compute_weights(train, test, args.boost, args.dist_frac, args.angle_deg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(weights, indent=1))
    vals = np.array(list(weights.values()))
    print(f"{args.out}: {len(weights)} weights, mean={vals.mean():.3f} max={vals.max():.3f} "
          f"boosted(>1)={int((vals > 1).sum())}")


if __name__ == "__main__":
    main()
