import csv

import numpy as np

from src.utils.pose_utils import (
    PoseSet, angular_gap_deg, coverage_stats, load_poses_from_csv,
    qvec2rotmat, scene_extent,
)


def _w2c_from_center_dir(center, yaw_deg):
    """Build a w2c (qvec-free) rotation looking along +x rotated by yaw about z."""
    yaw = np.radians(yaw_deg)
    z_cam = np.array([np.cos(yaw), np.sin(yaw), 0.0])  # viewing dir in world
    x_cam = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    y_cam = np.cross(z_cam, x_cam)
    R = np.stack([x_cam, y_cam, z_cam])  # rows = camera axes in world => w2c
    t = -R @ np.asarray(center, dtype=float)
    return R, t


def _rotmat2qvec(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [w, x, y, z]


def test_qvec_roundtrip():
    q = [0.7171, 0.0069, 0.3541, 0.6002]
    q = list(np.array(q) / np.linalg.norm(q))
    R = qvec2rotmat(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.allclose(_rotmat2qvec(R), q, atol=1e-6)


def test_center_and_dir_recovery(tmp_path):
    # Write a synthetic test_poses.csv and check center/dir recovery.
    poses = [((1.0, 2.0, 3.0), 0.0), ((-4.0, 0.5, 2.0), 90.0)]
    csv_path = tmp_path / "test_poses.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height".split(","))
        for i, (center, yaw) in enumerate(poses):
            R, t = _w2c_from_center_dir(center, yaw)
            q = _rotmat2qvec(R)
            w.writerow([f"img{i}.JPG", *q, *t, 926.0, 926.0, 660.0, 494.5, 1320, 989])

    ps = load_poses_from_csv(csv_path)
    assert ps.names == ["img0.JPG", "img1.JPG"]
    for i, (center, yaw) in enumerate(poses):
        assert np.allclose(ps.centers[i], center, atol=1e-6)
        expect_dir = [np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0.0]
        assert np.allclose(ps.view_dirs[i], expect_dir, atol=1e-6)


def test_coverage_stats_interp_vs_extrap():
    # Train ring: 10 cameras along x in [0,9], all looking the same way (+x).
    n = 10
    train = PoseSet(
        names=[f"t{i}" for i in range(n)],
        centers=np.stack([np.arange(n, dtype=float),
                          np.zeros(n), np.zeros(n)], axis=1),
        view_dirs=np.tile([1.0, 0.0, 0.0], (n, 1)),
    )
    # Query A sits between train cams, same dir -> interpolative.
    # Query B is far away with an opposite view dir -> extrapolative.
    query = PoseSet(
        names=["A", "B"],
        centers=np.array([[4.5, 0.0, 0.0], [50.0, 20.0, 0.0]]),
        view_dirs=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    )
    rows = coverage_stats(query, train, dist_frac_thresh=0.2, angle_thresh_deg=20)
    a, b = rows
    assert a["nearest_dist"] == 0.5 and a["angle_at_nearest"] < 1e-6
    assert a["n_near"] >= 2  # neighbors at x=4 and x=5 within 0.2*extent(=9)
    assert b["n_near"] == 0
    assert b["angle_at_nearest"] > 170
    assert b["nearest_dist_frac"] > 1.0


def test_scene_extent_and_angles():
    c = np.array([[0, 0, 0], [3, 4, 0.0]])
    assert scene_extent(c) == 5.0
    a = np.array([[1.0, 0, 0]])
    b = np.array([[0.0, 1.0, 0], [1.0, 0, 0]])
    ang = angular_gap_deg(a, b)
    assert np.allclose(ang, [[90.0, 0.0]], atol=1e-9)
