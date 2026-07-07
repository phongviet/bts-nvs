"""Week 2, init arm (b): dense COLMAP point-cloud init.

Pipeline (per scene), following docs/plan_week2.md Day 1:
    colmap image_undistorter -> colmap patch_match_stereo -> colmap stereo_fusion
    -> voxel-downsample fused.ply to a point budget
    -> synthesize a points3D.bin (dense points, no track info -- splatfacto/
       ColmapDataParser only needs xyz+rgb+error for init; max_2D_matches_per_3D_point
       defaults to 0 so the per-point 2D-track fields are never read)
    -> assemble train_staging_dense/{images,sparse/0} for ns-train

Requires: scene already filtered by filter_colmap_train.py (uses its
colmap_train_only/{cameras,images}.bin as the camera/pose source -- poses are
never re-solved, only the point cloud is swapped, per the plan's "never move
the test cameras" rule).
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d

from nerfstudio.data.utils.colmap_parsing_utils import (
    Point3D,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)


def run(cmd: list[str]):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def write_pruned_sparse_for_colmap_tools(train_only_dir: Path, out_dir: Path):
    """COLMAP's C++ image_undistorter/patch_match_stereo (unlike nerfstudio's
    ColmapDataParser) dereference every points3D track's image_ids -- but
    filter_colmap_train.py intentionally leaves points3D tracks referencing
    the filtered-out test/extra images (nerfstudio never reads them, so it's
    harmless there). Feeding that straight to colmap's mvs tools throws
    std::out_of_range in Model::ReadFromCOLMAP. Prune tracks down to only the
    kept images (dropping points left with zero observations) so colmap's own
    tools get a self-consistent model.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras = read_cameras_binary(train_only_dir / "cameras.bin")
    images = read_images_binary(train_only_dir / "images.bin")
    points3D = read_points3D_binary(train_only_dir / "points3D.bin")
    kept_image_ids = set(images.keys())

    pruned = {}
    for pid, pt in points3D.items():
        mask = np.array([iid in kept_image_ids for iid in pt.image_ids], dtype=bool)
        if not mask.any():
            continue
        pruned[pid] = pt._replace(
            image_ids=pt.image_ids[mask],
            point2D_idxs=pt.point2D_idxs[mask],
        )

    write_cameras_binary(cameras, out_dir / "cameras.bin")
    write_images_binary(images, out_dir / "images.bin")
    write_points3D_binary(pruned, out_dir / "points3D.bin")
    print(f"pruned points3D tracks: {len(points3D)} -> {len(pruned)} points with >=1 kept-image observation")


def build_dense_cloud(images_dir: Path, pruned_sparse_dir: Path, dense_dir: Path, max_image_size: int, gpu_index: str):
    dense_dir.mkdir(parents=True, exist_ok=True)
    if not (dense_dir / "fused.ply").exists():
        run([
            "colmap", "image_undistorter",
            "--image_path", str(images_dir),
            "--input_path", str(pruned_sparse_dir),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP",
            "--max_image_size", str(max_image_size),
        ])
        run([
            "colmap", "patch_match_stereo",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true",
            "--PatchMatchStereo.gpu_index", gpu_index,
            "--PatchMatchStereo.max_image_size", str(max_image_size),
        ])
        run([
            "colmap", "stereo_fusion",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", str(dense_dir / "fused.ply"),
        ])
    else:
        print(f"{dense_dir / 'fused.ply'} already exists, skipping COLMAP dense stereo.")


def voxel_downsample_to_budget(pcd: o3d.geometry.PointCloud, max_points: int) -> o3d.geometry.PointCloud:
    n = len(pcd.points)
    if n <= max_points:
        return pcd
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent())
    lo, hi = diag * 1e-5, diag * 0.1
    for _ in range(20):
        mid = (lo + hi) / 2
        down = pcd.voxel_down_sample(mid)
        if len(down.points) > max_points:
            lo = mid
        else:
            hi = mid
    return pcd.voxel_down_sample(hi)


def write_hybrid_sparse_model(colmap_train_only: Path, fused_ply: Path, out_dir: Path, max_points: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras = read_cameras_binary(colmap_train_only / "cameras.bin")
    images = read_images_binary(colmap_train_only / "images.bin")
    write_cameras_binary(cameras, out_dir / "cameras.bin")
    write_images_binary(images, out_dir / "images.bin")

    pcd = o3d.io.read_point_cloud(str(fused_ply))
    n_raw = len(pcd.points)
    pcd = voxel_downsample_to_budget(pcd, max_points)
    n_down = len(pcd.points)
    print(f"dense fused cloud: {n_raw} raw points -> {n_down} after voxel downsample (budget {max_points})")

    xyz = np.asarray(pcd.points, dtype=np.float64)
    if pcd.has_colors():
        rgb = (np.asarray(pcd.colors) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        rgb = np.full((n_down, 3), 128, dtype=np.uint8)

    empty_ids = np.array([], dtype=np.int64)
    points3D = {
        i: Point3D(
            id=i,
            xyz=xyz[i],
            rgb=rgb[i],
            error=np.float64(0.0),
            image_ids=empty_ids,
            point2D_idxs=empty_ids,
        )
        for i in range(n_down)
    }
    write_points3D_binary(points3D, out_dir / "points3D.bin")
    return n_raw, n_down


def make_train_dir(scene_dir: Path, sparse_dir: Path, staging_dir: Path):
    staging_dir.mkdir(parents=True, exist_ok=True)
    images_link = staging_dir / "images"
    sparse_link = staging_dir / "sparse" / "0"
    if images_link.exists() or images_link.is_symlink():
        images_link.unlink()
    if sparse_link.exists() or sparse_link.is_symlink():
        sparse_link.unlink()
    sparse_link.parent.mkdir(parents=True, exist_ok=True)
    images_link.symlink_to((scene_dir / "train" / "images").resolve())
    sparse_link.symlink_to(sparse_dir.resolve())
    print(f"Dense training data dir ready: {staging_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path, help="e.g. data/raw/phase1/public_set/hcm0034")
    ap.add_argument("--processed-root", required=True, type=Path, help="e.g. data/processed/phase1")
    ap.add_argument("--max-points", type=int, default=2_000_000)
    ap.add_argument("--max-image-size", type=int, default=1600, help="cap for undistort/patch-match (VRAM budget)")
    ap.add_argument("--gpu-index", default="0")
    args = ap.parse_args()

    scene = args.scene_dir.name
    scene_root = args.processed_root / scene
    train_only = scene_root / "colmap_train_only"
    staging = scene_root / "train_staging"
    if not (train_only / "images.bin").exists():
        raise RuntimeError(f"{train_only} missing -- run filter_colmap_train.py for {scene} first.")

    pruned_sparse = scene_root / "colmap_train_only_pruned_for_mvs"
    write_pruned_sparse_for_colmap_tools(train_only, pruned_sparse)

    dense_dir = scene_root / "dense"
    build_dense_cloud(staging / "images", pruned_sparse, dense_dir, args.max_image_size, args.gpu_index)

    hybrid_sparse = scene_root / "colmap_dense_init"
    n_raw, n_down = write_hybrid_sparse_model(train_only, dense_dir / "fused.ply", hybrid_sparse, args.max_points)

    dense_staging = scene_root / "train_staging_dense"
    make_train_dir(args.scene_dir, hybrid_sparse, dense_staging)

    print(f"{scene}: dense init ready. raw_points={n_raw} downsampled_points={n_down} -> {dense_staging}")


if __name__ == "__main__":
    main()
