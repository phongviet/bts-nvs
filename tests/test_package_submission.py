import csv
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.package_submission import package, validate_renders, validate_zip

W, H = 96, 64


def _make_scene(root: Path, scene: str, names, runs: Path, good=True):
    test_dir = root / scene / "test"
    test_dir.mkdir(parents=True)
    with open(test_dir / "test_poses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height".split(","))
        for n in names:
            w.writerow([n, 1, 0, 0, 0, 0, 0, 0, 90.0, 90.0, W / 2, H / 2, W, H])
    render_dir = runs / scene / "renders_test"
    render_dir.mkdir(parents=True)
    if good:
        for n in names:
            Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)).save(render_dir / n, quality=95)
    return render_dir


def test_validate_ok_and_package(tmp_path):
    poses_root = tmp_path / "raw"
    runs = tmp_path / "runs"
    _make_scene(poses_root, "SC1", ["a.JPG", "b.JPG"], runs)
    (runs / "SC1" / "renders_test" / ".done").touch()  # marker must be ignored
    assert validate_renders(runs, ["SC1"], poses_root) == []

    out = tmp_path / "sub" / "submission_round1.zip"
    package(runs, ["SC1"], out, poses_root=poses_root)
    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        assert sorted(zf.namelist()) == ["SC1/a.JPG", "SC1/b.JPG"]
    assert validate_zip(out, ["SC1"], poses_root) == []


def test_missing_image_fails(tmp_path):
    poses_root, runs = tmp_path / "raw", tmp_path / "runs"
    rd = _make_scene(poses_root, "SC1", ["a.JPG", "b.JPG"], runs)
    (rd / "b.JPG").unlink()
    errors = validate_renders(runs, ["SC1"], poses_root)
    assert any("missing" in e for e in errors)
    with pytest.raises(RuntimeError):
        package(runs, ["SC1"], tmp_path / "out.zip", poses_root=poses_root)


def test_extra_and_size_mismatch_fail(tmp_path):
    poses_root, runs = tmp_path / "raw", tmp_path / "runs"
    rd = _make_scene(poses_root, "SC1", ["a.JPG"], runs)
    Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)).save(rd / "zz.JPG")  # extra
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(rd / "a.JPG")  # wrong size
    errors = validate_renders(runs, ["SC1"], poses_root)
    assert any("extra" in e for e in errors)
    assert any("size" in e for e in errors)


def test_png_bytes_in_jpg_name_fails(tmp_path):
    poses_root, runs = tmp_path / "raw", tmp_path / "runs"
    rd = _make_scene(poses_root, "SC1", ["a.JPG"], runs)
    Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)).save(rd / "a.JPG", format="PNG")
    errors = validate_renders(runs, ["SC1"], poses_root)
    assert any("PNG bytes" in e for e in errors)


def test_undecodable_fails(tmp_path):
    poses_root, runs = tmp_path / "raw", tmp_path / "runs"
    rd = _make_scene(poses_root, "SC1", ["a.JPG"], runs)
    (rd / "a.JPG").write_bytes(b"not an image at all")
    errors = validate_renders(runs, ["SC1"], poses_root)
    assert any("undecodable" in e for e in errors)


def test_zip_missing_scene_fails(tmp_path):
    poses_root, runs = tmp_path / "raw", tmp_path / "runs"
    _make_scene(poses_root, "SC1", ["a.JPG"], runs)
    _make_scene(poses_root, "SC2", ["a.JPG"], runs)
    out = tmp_path / "sub.zip"
    package(runs, ["SC1"], out, poses_root=poses_root)  # deliberately only SC1
    errors = validate_zip(out, ["SC1", "SC2"], poses_root)
    assert any("missing scenes" in e for e in errors)
