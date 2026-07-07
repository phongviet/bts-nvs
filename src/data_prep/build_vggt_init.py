"""Week 2, init arm (c): VGGT-style vision-only pseudo-point-cloud init.

Per docs/plan_week2.md Day 2: run a generic, scene-agnostic feed-forward
geometry model (VGGT, Meta AI, generically pretrained -- see
docs/rules_and_constraints.md for provenance) on the scene's train images to
get per-image depth + predicted camera poses, Umeyama-align the predicted
camera centers onto our *real* COLMAP camera centers (never re-solve the
actual poses), unproject the scale-corrected depth into the COLMAP world
frame, and aggregate into a dense pseudo point cloud. This attacks a
different failure mode than dense COLMAP: surfaces SfM structurally can't
triangulate (thin lattice, textureless metal/sky), rather than "too few
points".

VGGT processes all input frames jointly in one attention pass, so its
predicted scale/frame is only self-consistent *within one forward call* --
we chunk the scene's images and fit a *separate* similarity (Umeyama)
transform per chunk from that chunk's own predicted-vs-real camera centers,
then aggregate all chunks' now-aligned points. Do not pool correspondences
across chunks into one global fit -- different chunks have no shared gauge.
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from nerfstudio.data.utils.colmap_parsing_utils import (
    Point3D,
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def umeyama_alignment(src: np.ndarray, dst: np.ndarray):
    """Similarity transform (scale, R, t) mapping src -> dst (both Nx3), least squares."""
    assert src.shape == dst.shape and src.shape[0] >= 3
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = (dst_c.T @ src_c) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / src.shape[0]
    scale = np.trace(np.diag(D) @ S) / var_src
    t = mu_dst - scale * R @ mu_src
    return scale, R, t


def colmap_camera_centers(images_bin: dict, names: list[str]) -> np.ndarray:
    name_to_img = {im.name: im for im in images_bin.values()}
    centers = []
    for n in names:
        im = name_to_img[n]
        R = qvec2rotmat(im.qvec)
        t = im.tvec
        centers.append(-R.T @ t)
    return np.stack(centers)


def run_vggt_chunk(model, device, dtype, image_paths: list[str]):
    images = load_and_preprocess_images(image_paths, mode="pad")
    images = images.to(device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype):
            batch = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(batch)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batch.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, batch, ps_idx)
    extrinsic = extrinsic.squeeze(0).float().cpu().numpy()
    intrinsic = intrinsic.squeeze(0).float().cpu().numpy()
    depth_map = depth_map.squeeze(0).float().cpu().numpy()
    depth_conf = depth_conf.squeeze(0).float().cpu().numpy()
    images_np = (images.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)  # (S,3,H,W)
    return extrinsic, intrinsic, depth_map, depth_conf, images_np


def points_from_chunk(extrinsic, intrinsic, depth_map, depth_conf, images_np,
                       conf_percentile: float, points_per_frame_cap: int):
    world_pts = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)  # (S,H,W,3)
    S, H, W = depth_map.shape[:3]
    all_xyz, all_rgb = [], []
    for s in range(S):
        conf = depth_conf[s]
        thresh = np.percentile(conf, conf_percentile)
        mask = conf >= max(thresh, 1e-6)
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        if len(ys) > points_per_frame_cap:
            idx = np.random.choice(len(ys), points_per_frame_cap, replace=False)
            ys, xs = ys[idx], xs[idx]
        xyz = world_pts[s, ys, xs]
        rgb = images_np[s][:, ys, xs].T  # (N,3)
        all_xyz.append(xyz)
        all_rgb.append(rgb)
    if not all_xyz:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(all_xyz), np.concatenate(all_rgb)


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
    print(f"VGGT training data dir ready: {staging_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path)
    ap.add_argument("--processed-root", required=True, type=Path)
    ap.add_argument("--chunk-size", type=int, default=16, help="images per VGGT forward pass (VRAM budget)")
    ap.add_argument("--conf-percentile", type=float, default=50.0, help="per-frame confidence keep threshold")
    ap.add_argument("--points-per-frame-cap", type=int, default=8000)
    ap.add_argument("--max-points", type=int, default=2_000_000)
    ap.add_argument("--min-chunk-for-umeyama", type=int, default=4)
    args = ap.parse_args()

    scene = args.scene_dir.name
    scene_root = args.processed_root / scene
    train_only = scene_root / "colmap_train_only"
    if not (train_only / "images.bin").exists():
        raise RuntimeError(f"{train_only} missing -- run filter_colmap_train.py for {scene} first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading VGGT-1B on {device} ({dtype}) ...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

    images_bin = read_images_binary(train_only / "images.bin")
    image_names = sorted(im.name for im in images_bin.values())
    image_paths = [str(args.scene_dir / "train" / "images" / n) for n in image_names]

    chunks = [image_names[i:i + args.chunk_size] for i in range(0, len(image_names), args.chunk_size)]
    all_xyz, all_rgb = [], []
    skipped_chunks = 0
    for ci, chunk_names in enumerate(chunks):
        chunk_paths = [str(args.scene_dir / "train" / "images" / n) for n in chunk_names]
        print(f"[{scene}] VGGT chunk {ci + 1}/{len(chunks)} ({len(chunk_names)} images)")
        extrinsic, intrinsic, depth_map, depth_conf, images_np = run_vggt_chunk(model, device, dtype, chunk_paths)

        pred_centers = np.stack([-extrinsic[s, :3, :3].T @ extrinsic[s, :3, 3] for s in range(len(chunk_names))])
        real_centers = colmap_camera_centers(images_bin, chunk_names)

        if len(chunk_names) < args.min_chunk_for_umeyama:
            print(f"  chunk too small ({len(chunk_names)}) for a stable Umeyama fit, skipping")
            skipped_chunks += 1
            continue
        scale, R, t = umeyama_alignment(pred_centers, real_centers)
        resid = np.linalg.norm((scale * (R @ pred_centers.T).T + t) - real_centers, axis=1)
        print(f"  Umeyama fit: scale={scale:.4f} mean_center_resid={resid.mean():.4f} max={resid.max():.4f}")

        xyz, rgb = points_from_chunk(extrinsic, intrinsic, depth_map, depth_conf, images_np,
                                      args.conf_percentile, args.points_per_frame_cap)
        if len(xyz) == 0:
            continue
        xyz_aligned = (scale * (R @ xyz.T).T) + t
        all_xyz.append(xyz_aligned)
        all_rgb.append(rgb)

    if not all_xyz:
        raise RuntimeError(f"{scene}: no VGGT points produced (all chunks skipped/empty)")

    xyz = np.concatenate(all_xyz)
    rgb = np.concatenate(all_rgb)
    print(f"{scene}: {len(xyz)} raw aligned VGGT points from {len(chunks) - skipped_chunks}/{len(chunks)} usable chunks")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    pcd = voxel_downsample_to_budget(pcd, args.max_points)
    n_down = len(pcd.points)
    print(f"{scene}: downsampled to {n_down} points (budget {args.max_points})")

    out_ply = scene_root / "vggt" / "points.ply"
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_ply), pcd)

    cameras = read_cameras_binary(train_only / "cameras.bin")
    hybrid_sparse = scene_root / "colmap_vggt_init"
    hybrid_sparse.mkdir(parents=True, exist_ok=True)
    write_cameras_binary(cameras, hybrid_sparse / "cameras.bin")
    write_images_binary(images_bin, hybrid_sparse / "images.bin")

    xyz_d = np.asarray(pcd.points)
    rgb_d = (np.asarray(pcd.colors) * 255.0).clip(0, 255).astype(np.uint8)
    empty_ids = np.array([], dtype=np.int64)
    points3D = {
        i: Point3D(id=i, xyz=xyz_d[i], rgb=rgb_d[i], error=np.float64(0.0),
                   image_ids=empty_ids, point2D_idxs=empty_ids)
        for i in range(n_down)
    }
    write_points3D_binary(points3D, hybrid_sparse / "points3D.bin")

    vggt_staging = scene_root / "train_staging_vggt"
    make_train_dir(args.scene_dir, hybrid_sparse, vggt_staging)
    print(f"{scene}: VGGT init ready. points={n_down} -> {vggt_staging}")


if __name__ == "__main__":
    main()
