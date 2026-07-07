import numpy as np
import pytest
from PIL import Image

from src.ensemble_renders import ensemble


def _dir_with(tmp_path, name, value, filenames=("a.JPG", "b.JPG")):
    d = tmp_path / name
    d.mkdir()
    for fn in filenames:
        Image.fromarray(np.full((16, 24, 3), value, dtype=np.uint8)).save(d / fn, quality=100)
    return d


def test_mean_of_two(tmp_path):
    d1 = _dir_with(tmp_path, "r1", 100)
    d2 = _dir_with(tmp_path, "r2", 200)
    out = tmp_path / "out"
    n = ensemble([d1, d2], out, "mean", encoder="png")
    assert n == 2
    arr = np.asarray(Image.open(out / "a.JPG").convert("RGB"))
    assert np.all(arr == 150)


def test_median_of_three_rejects_outlier(tmp_path):
    dirs = [_dir_with(tmp_path, f"r{v}", v) for v in (100, 102, 250)]
    out = tmp_path / "out"
    ensemble(dirs, out, "median", encoder="png")
    arr = np.asarray(Image.open(out / "a.JPG").convert("RGB"))
    assert np.all(arr == 102)


def test_filename_mismatch_fails(tmp_path):
    d1 = _dir_with(tmp_path, "r1", 100)
    d2 = _dir_with(tmp_path, "r2", 100, filenames=("a.JPG", "c.JPG"))
    with pytest.raises(SystemExit):
        ensemble([d1, d2], tmp_path / "out", "mean")


def test_median_needs_three(tmp_path):
    d1 = _dir_with(tmp_path, "r1", 100)
    d2 = _dir_with(tmp_path, "r2", 100)
    with pytest.raises(SystemExit):
        ensemble([d1, d2], tmp_path / "out", "median")
