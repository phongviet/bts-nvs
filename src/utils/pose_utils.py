"""Shared COLMAP-convention pose helpers for coverage/val-split/weighting tools.

All poses in this codebase (test_poses.csv AND images.bin) are COLMAP
world-to-camera: x_cam = R(q) @ x_world + t. So:
    camera center (world)   C = -R^T t
    viewing direction (world, cam +z) = R^T e_z = third row of R
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PoseSet:
    names: list[str]        # image filenames
    centers: np.ndarray     # (N,3) camera centers, world
    view_dirs: np.ndarray   # (N,3) unit viewing directions, world

    def __len__(self) -> int:
        return len(self.names)


def qvec2rotmat(qvec) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def _center_and_dir(qvec, tvec) -> tuple[np.ndarray, np.ndarray]:
    R = qvec2rotmat(qvec)
    t = np.asarray(tvec, dtype=np.float64)
    center = -R.T @ t
    view_dir = R[2, :]  # R^T @ [0,0,1]
    return center, view_dir / np.linalg.norm(view_dir)


def load_poses_from_csv(csv_path: Path) -> PoseSet:
    """test_poses.csv schema: image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height"""
    names, centers, dirs = [], [], []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            c, d = _center_and_dir(
                [float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])],
                [float(r["tx"]), float(r["ty"]), float(r["tz"])],
            )
            names.append(r["image_name"])
            centers.append(c)
            dirs.append(d)
    return PoseSet(names, np.array(centers), np.array(dirs))


def load_poses_from_colmap(sparse_dir: Path) -> PoseSet:
    from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary

    images = read_images_binary(Path(sparse_dir) / "images.bin")
    names, centers, dirs = [], [], []
    for _, im in sorted(images.items(), key=lambda kv: kv[1].name):
        c, d = _center_and_dir(im.qvec, im.tvec)
        names.append(im.name)
        centers.append(c)
        dirs.append(d)
    return PoseSet(names, np.array(centers), np.array(dirs))


def scene_extent(centers: np.ndarray) -> float:
    """Diagonal of the camera-center bounding box -- the normalizer for distances."""
    return float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))


def angular_gap_deg(dirs_a: np.ndarray, dirs_b: np.ndarray) -> np.ndarray:
    """(A,3) x (B,3) unit dirs -> (A,B) pairwise angles in degrees."""
    cos = np.clip(dirs_a @ dirs_b.T, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def coverage_stats(query: PoseSet, ref: PoseSet,
                   dist_frac_thresh: float = 0.05,
                   angle_thresh_deg: float = 20.0) -> list[dict]:
    """Per query pose vs the reference (train) set:
      nearest_dist       distance to nearest ref center (world units)
      nearest_dist_frac  same, normalized by ref scene extent
      angle_at_nearest   angular gap (deg) to that nearest-position ref view
      min_angle          smallest angular gap to ANY ref view
      n_near             ref views with dist_frac < dist_frac_thresh AND angle < angle_thresh_deg
      n_near_loose       same with both thresholds doubled (robust coverage check)
    """
    extent = scene_extent(ref.centers)
    d = np.linalg.norm(query.centers[:, None, :] - ref.centers[None, :, :], axis=2)  # (Q,R)
    ang = angular_gap_deg(query.view_dirs, ref.view_dirs)                            # (Q,R)
    nearest = d.argmin(axis=1)
    near_mask = (d / extent < dist_frac_thresh) & (ang < angle_thresh_deg)
    near_loose = (d / extent < 2 * dist_frac_thresh) & (ang < 2 * angle_thresh_deg)

    rows = []
    for i, name in enumerate(query.names):
        rows.append({
            "image_name": name,
            "nearest_dist": float(d[i, nearest[i]]),
            "nearest_dist_frac": float(d[i, nearest[i]] / extent),
            "nearest_train": ref.names[nearest[i]],
            "angle_at_nearest": float(ang[i, nearest[i]]),
            "min_angle": float(ang[i].min()),
            "n_near": int(near_mask[i].sum()),
            "n_near_loose": int(near_loose[i].sum()),
        })
    return rows
