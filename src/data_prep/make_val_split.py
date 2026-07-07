"""Cut a local validation split from the train images, for scenes where test
GT is not available (private_set1).

Two modes:
  nth         hold out every Nth train image (Mip-NeRF360 convention) --
              the original Week-1 behavior.
  match-test  (A.6b) choose hold-outs by proximity (position + viewing angle)
              to that scene's test_poses.csv, so the val split has the same
              pose distribution the graded score is computed on. For each of
              n_val evenly-subsampled test poses, pick the nearest unused
              train image under cost = dist/scene_extent + angle_deg/45.

Writes <out-dir>/{train_ids.txt,val_ids.txt} (+ val_match.csv in match-test
mode: which test pose each val image stands in for, with the match cost).

Usage (match-test, private scene):
  python src/data_prep/make_val_split.py --mode match-test \
      --images-dir data/raw/phase1/private_set1/HCM0249/train/images \
      --sparse-dir data/processed/phase1/HCM0249/colmap_train_only \
      --test-poses data/raw/phase1/private_set1/HCM0249/test/test_poses.csv \
      --out-dir data/processed/phase1/HCM0249/splits --n-val 30
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.pose_utils import (  # noqa: E402
    angular_gap_deg, load_poses_from_colmap, load_poses_from_csv, scene_extent,
)


def list_images(images_dir: Path) -> list[str]:
    return sorted(p.name for p in images_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def make_split_nth(images_dir: Path, out_dir: Path, every_n: int = 8):
    names = list_images(images_dir)
    val = names[::every_n]
    train = [n for n in names if n not in set(val)]
    write_split(out_dir, train, val)
    print(f"{images_dir}: {len(train)} train / {len(val)} val (every {every_n})")


def match_test_split(train_poses, test_poses, n_val: int, angle_norm_deg: float = 45.0):
    """Return (val_indices_into_train, match_rows). Greedy nearest-unused-train
    assignment over n_val evenly-subsampled test poses."""
    extent = scene_extent(train_poses.centers)
    d = np.linalg.norm(test_poses.centers[:, None, :] - train_poses.centers[None, :, :], axis=2)
    ang = angular_gap_deg(test_poses.view_dirs, train_poses.view_dirs)
    cost = d / extent + ang / angle_norm_deg  # (T, N_train)

    n_val = min(n_val, len(train_poses), len(test_poses))
    target_idx = np.linspace(0, len(test_poses) - 1, n_val).round().astype(int)
    target_idx = sorted(set(target_idx.tolist()))

    used: set[int] = set()
    val_indices, match_rows = [], []
    # Assign hardest-to-cover targets first so greedy doesn't steal their only neighbor.
    order = sorted(target_idx, key=lambda t: float(np.min(cost[t])), reverse=True)
    for t in order:
        ranked = np.argsort(cost[t])
        pick = next(int(i) for i in ranked if int(i) not in used)
        used.add(pick)
        val_indices.append(pick)
        match_rows.append({
            "val_image": train_poses.names[pick],
            "matched_test_pose": test_poses.names[t],
            "cost": round(float(cost[t, pick]), 4),
            "dist_frac": round(float(d[t, pick] / extent), 5),
            "angle_deg": round(float(ang[t, pick]), 2),
        })
    return val_indices, sorted(match_rows, key=lambda r: r["val_image"])


def make_split_match_test(images_dir: Path, sparse_dir: Path, test_poses_csv: Path,
                          out_dir: Path, n_val: int):
    train_poses = load_poses_from_colmap(sparse_dir)
    test_poses = load_poses_from_csv(test_poses_csv)
    names_on_disk = set(list_images(images_dir))
    # keep only poses whose image actually exists in train/images
    keep = [i for i, n in enumerate(train_poses.names) if n in names_on_disk]
    train_poses.names = [train_poses.names[i] for i in keep]
    train_poses.centers = train_poses.centers[keep]
    train_poses.view_dirs = train_poses.view_dirs[keep]

    val_idx, match_rows = match_test_split(train_poses, test_poses, n_val)
    val = sorted(train_poses.names[i] for i in val_idx)
    train = [n for n in sorted(names_on_disk) if n not in set(val)]
    write_split(out_dir, train, val)

    with open(out_dir / "val_match.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(match_rows[0].keys()))
        w.writeheader()
        w.writerows(match_rows)
    costs = [r["cost"] for r in match_rows]
    print(f"{images_dir.parent.parent.name}: {len(train)} train / {len(val)} val "
          f"(match-test, n_val={len(val)}, cost p50={np.median(costs):.3f} max={max(costs):.3f})")


def write_split(out_dir: Path, train: list[str], val: list[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "val_ids.txt").write_text("\n".join(val) + "\n")
    (out_dir / "train_ids.txt").write_text("\n".join(train) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["nth", "match-test"], default="nth")
    ap.add_argument("--images-dir", required=True, type=Path, help="scene train/images dir")
    ap.add_argument("--out-dir", required=True, type=Path, help="scene splits output dir")
    ap.add_argument("--every-n", type=int, default=8, help="(nth mode)")
    ap.add_argument("--sparse-dir", type=Path, help="(match-test) colmap_train_only dir")
    ap.add_argument("--test-poses", type=Path, help="(match-test) test_poses.csv")
    ap.add_argument("--n-val", type=int, default=30, help="(match-test) val set size")
    args = ap.parse_args()

    if args.mode == "nth":
        make_split_nth(args.images_dir, args.out_dir, args.every_n)
    else:
        if not (args.sparse_dir and args.test_poses):
            ap.error("--sparse-dir and --test-poses are required in match-test mode")
        make_split_match_test(args.images_dir, args.sparse_dir, args.test_poses,
                              args.out_dir, args.n_val)


if __name__ == "__main__":
    main()
