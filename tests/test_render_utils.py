import struct

import numpy as np
import pytest

from src.utils.render_utils import load_colmap_poses


def _write_cameras_bin(path, cam_id=1, model_id=1, w=1320, h=989,
                       params=(1200.0, 1200.0, 660.0, 494.5)):
    # model_id 1 = PINHOLE (4 params)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", cam_id, model_id, w, h))
        for p in params:
            f.write(struct.pack("<d", p))


def _write_images_bin(path, names, cam_id=1):
    rng = np.random.default_rng(0)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(names)))
        for i, name in enumerate(names):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            t = rng.normal(size=3)
            f.write(struct.pack("<i", i + 1))
            f.write(struct.pack("<dddd", *q))
            f.write(struct.pack("<ddd", *t))
            f.write(struct.pack("<i", cam_id))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))  # no 2D points


@pytest.fixture
def sparse_dir(tmp_path):
    names = ["img_003.jpg", "img_001.jpg", "img_002.jpg"]
    _write_cameras_bin(tmp_path / "cameras.bin")
    _write_images_bin(tmp_path / "images.bin", names)
    return tmp_path


def test_load_all_sorted_by_name(sparse_dir):
    rows = load_colmap_poses(sparse_dir)
    assert [r["image_name"] for r in rows] == ["img_001.jpg", "img_002.jpg", "img_003.jpg"]
    r = rows[0]
    assert r["fx"] == 1200.0 and r["fy"] == 1200.0
    assert r["width"] == 1320 and r["height"] == 989
    assert len(r["qvec"]) == 4 and len(r["tvec"]) == 3


def test_only_names_filter(sparse_dir):
    rows = load_colmap_poses(sparse_dir, only_names={"img_002.jpg"})
    assert [r["image_name"] for r in rows] == ["img_002.jpg"]


def test_only_names_missing_gives_empty(sparse_dir):
    assert load_colmap_poses(sparse_dir, only_names={"nope.jpg"}) == []


def test_simple_radial_camera_shares_focal(tmp_path):
    # model_id 2 = SIMPLE_RADIAL (f, cx, cy, k)
    _write_cameras_bin(tmp_path / "cameras.bin", model_id=2,
                       params=(1100.0, 660.0, 494.5, 0.01))
    _write_images_bin(tmp_path / "images.bin", ["a.jpg"])
    rows = load_colmap_poses(tmp_path)
    assert rows[0]["fx"] == 1100.0 and rows[0]["fy"] == 1100.0
