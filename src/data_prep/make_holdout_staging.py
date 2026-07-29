"""Build a backbone-training staging dir that EXCLUDES a val-split's hold-out
frames, so a match-test val score measures held-out generalization (not
memorized views).

Why this exists: round-2 indoor scenes (bonsai, chair) ship no test GT and have
no same-regime public bench, so the only way to score them locally is a
match-test hold-out split (make_val_split.py). But render_val is only honest if
the backbone never trained on the val frames. The existing
`train_staging_dense` symlinks ALL train images + a sparse model registering all
of them, so a backbone trained on it has seen every val frame. This script
writes a sibling staging that keeps only `train_ids.txt` frames:

  <staging>/images/            real dir, symlinks to train_ids frames only
  <staging>/sparse/0/cameras.bin      copied verbatim
  <staging>/sparse/0/points3D.bin     copied verbatim  (see leak note below)
  <staging>/sparse/0/images.bin       filtered to train_ids only

Leak note: points3D.bin is the DENSE fused cloud, built by MVS that saw the val
views, so the geometry INIT is mildly informed by val. This is deliberate and
conservative: the photometric loss (what LPIPS/SSIM actually measure) strictly
excludes val, and a slightly better-initialised backbone at val poses only makes
the downstream "does DIBR/refiner help" delta HARDER to show, never easier.
Rebuilding dense sans-val would cost another full MVS pass for no decision value.

Usage:
  python src/data_prep/make_holdout_staging.py \
      --src-staging data/processed/round2/bonsai/train_staging_dense \
      --split-dir  data/processed/round2/bonsai/splits \
      --out-staging data/processed/round2/bonsai/train_staging_holdout
"""
import argparse
import shutil
from pathlib import Path

from nerfstudio.data.utils.colmap_parsing_utils import (
    read_images_binary,
    write_images_binary,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-staging", required=True, type=Path,
                    help="existing full-train staging (images/ + sparse/0/)")
    ap.add_argument("--split-dir", required=True, type=Path,
                    help="dir holding train_ids.txt / val_ids.txt")
    ap.add_argument("--out-staging", required=True, type=Path)
    args = ap.parse_args()

    train_ids = set((args.split_dir / "train_ids.txt").read_text().split())
    val_ids = set((args.split_dir / "val_ids.txt").read_text().split())
    if not train_ids:
        raise SystemExit(f"empty train_ids.txt in {args.split_dir}")

    src_sparse = args.src_staging / "sparse" / "0"
    src_images = (args.src_staging / "images").resolve()  # follows the symlink

    # --- filter images.bin to train_ids only ---
    images = read_images_binary(src_sparse / "images.bin")
    kept = {k: v for k, v in images.items() if v.name in train_ids}
    kept_names = {v.name for v in kept.values()}
    missing = train_ids - kept_names
    if missing:
        raise SystemExit(f"{len(missing)} train_ids have no pose in images.bin: "
                         f"{sorted(missing)[:5]}...")
    leaked = val_ids & kept_names
    if leaked:
        raise SystemExit(f"BUG: {len(leaked)} val frames survived the filter: "
                         f"{sorted(leaked)[:5]}...")

    out_sparse = args.out_staging / "sparse" / "0"
    out_sparse.mkdir(parents=True, exist_ok=True)
    write_images_binary(kept, out_sparse / "images.bin")
    for f in ("cameras.bin", "points3D.bin"):
        shutil.copyfile(src_sparse / f, out_sparse / f)

    # --- images/ dir with train_ids symlinks only (NOT a whole-dir symlink) ---
    out_images = args.out_staging / "images"
    if out_images.exists() or out_images.is_symlink():
        if out_images.is_dir() and not out_images.is_symlink():
            shutil.rmtree(out_images)
        else:
            out_images.unlink()
    out_images.mkdir(parents=True)
    n_linked = 0
    for name in sorted(train_ids):
        target = src_images / name
        if not target.exists():
            raise SystemExit(f"train image not found on disk: {target}")
        (out_images / name).symlink_to(target)
        n_linked += 1

    print(f"hold-out staging: {len(kept)} posed / {n_linked} images "
          f"(excluded {len(val_ids)} val) -> {args.out_staging}")


if __name__ == "__main__":
    main()
