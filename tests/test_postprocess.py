import numpy as np
from PIL import Image

from src.postprocess.ops import ENCODERS, OPS, process_dir, process_image


def _write_test_image(path, seed=0):
    rng = np.random.default_rng(seed)
    # low-res noise upscaled = smooth gradients, so JPEG behaves like a photo
    base = rng.integers(0, 255, size=(8, 12, 3), dtype=np.uint8)
    img = Image.fromarray(base).resize((96, 64), Image.BILINEAR)
    img.save(path, quality=100)
    return np.asarray(Image.open(path).convert("RGB"))


def test_all_ops_preserve_shape_and_range(tmp_path):
    src = tmp_path / "a.JPG"
    _write_test_image(src)
    for op_name in OPS:
        dst = tmp_path / op_name / "a.JPG"
        process_image(src, dst, op_name, "jpeg95")
        out = np.asarray(Image.open(dst).convert("RGB"))
        assert out.shape == (64, 96, 3), op_name
        assert out.dtype == np.uint8


def test_identity_jpeg_quality_ladder(tmp_path):
    """Higher JPEG quality must round-trip closer to the source."""
    src = tmp_path / "a.JPG"
    ref = _write_test_image(src)
    errs = {}
    for enc in ["jpeg75", "jpeg95", "png"]:
        dst = tmp_path / enc / "a.JPG"
        process_image(src, dst, "identity", enc)
        out = np.asarray(Image.open(dst).convert("RGB"))
        errs[enc] = float(np.mean((out.astype(float) - ref.astype(float)) ** 2))
    assert errs["png"] == 0.0
    assert errs["jpeg95"] < errs["jpeg75"]


def test_process_dir_keeps_names(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    for name in ["DJI_0001_V.JPG", "DJI_0002_V.JPG"]:
        _write_test_image(src_dir / name)
    n = process_dir(src_dir, tmp_path / "dst", "unsharp_r1_p50", "jpeg95")
    assert n == 2
    assert sorted(p.name for p in (tmp_path / "dst").iterdir()) == [
        "DJI_0001_V.JPG", "DJI_0002_V.JPG"]


def test_encoder_table_sanity():
    assert set(ENCODERS) >= {"jpeg75", "jpeg90", "jpeg95", "jpeg98", "png"}
    assert ENCODERS["jpeg75"]["quality"] == 75  # what render.py ships today
