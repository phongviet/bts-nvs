"""Filter a scene's COLMAP sparse model down to only the images physically
present in train/images/.

Why this exists: the provided sparse/0/images.bin contains poses for the
FULL capture (train + test + extra registration-only frames -- e.g. for
hcm0034: 337 posed images = 240 train + 60 test + 37 extras). Nerfstudio's
default ColmapDataParser does its own eval_interval split over ALL images
referenced in images.bin, which would silently leak the 60 held-out test
images (and 37 unknowns) into the "train" split. This script writes a
filtered sparse model containing only the poses whose image name exists in
train/images/, so training never sees test pixels.

Output: data/processed/<phase>/<scene>/colmap_train_only/{cameras.bin,images.bin,points3D.bin}
Point ns-train at a directory with images/ -> train/images and
sparse/0 -> this filtered output (see make_train_dir below).
"""
import argparse
import shutil
from pathlib import Path

from nerfstudio.data.utils.colmap_parsing_utils import (
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)


def filter_scene(scene_dir: Path, out_dir: Path):
    train_images_dir = scene_dir / "train" / "images"
    sparse_dir = scene_dir / "train" / "sparse" / "0"
    train_files = {p.name for p in train_images_dir.iterdir()}

    images = read_images_binary(sparse_dir / "images.bin")
    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    points3D = read_points3D_binary(sparse_dir / "points3D.bin")

    kept = {k: v for k, v in images.items() if v.name in train_files}
    missing = train_files - {v.name for v in kept.values()}
    if missing:
        raise RuntimeError(f"{len(missing)} train images have no pose in images.bin: {sorted(missing)[:5]}...")

    # Drop point3D observations that only reference filtered-out images (optional but keeps model consistent);
    # simplest safe approach: keep points3D as-is (image_ids referencing removed images are just unused for training).
    out_dir.mkdir(parents=True, exist_ok=True)
    write_images_binary(kept, out_dir / "images.bin")
    write_cameras_binary(cameras, out_dir / "cameras.bin")
    write_points3D_binary(points3D, out_dir / "points3D.bin")
    print(f"{scene_dir.name}: kept {len(kept)}/{len(images)} posed images (train dir has {len(train_files)} files) -> {out_dir}")


def make_train_dir(scene_dir: Path, filtered_sparse_dir: Path, staging_dir: Path):
    """Build a directory ns-train can point at: images/ + sparse/0/ (filtered)."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    images_link = staging_dir / "images"
    sparse_link = staging_dir / "sparse" / "0"
    if images_link.exists() or images_link.is_symlink():
        images_link.unlink()
    if sparse_link.exists() or sparse_link.is_symlink():
        shutil.rmtree(sparse_link) if sparse_link.is_dir() and not sparse_link.is_symlink() else sparse_link.unlink()
    sparse_link.parent.mkdir(parents=True, exist_ok=True)

    images_link.symlink_to((scene_dir / "train" / "images").resolve())
    sparse_link.symlink_to(filtered_sparse_dir.resolve())
    print(f"Training data dir ready: {staging_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path, help="e.g. data/raw/phase1/public_set/hcm0034")
    ap.add_argument("--processed-root", required=True, type=Path, help="e.g. data/processed/phase1")
    args = ap.parse_args()

    scene = args.scene_dir.name
    filtered_sparse = args.processed_root / scene / "colmap_train_only"
    filter_scene(args.scene_dir, filtered_sparse)

    staging = args.processed_root / scene / "train_staging"
    make_train_dir(args.scene_dir, filtered_sparse, staging)


if __name__ == "__main__":
    main()
