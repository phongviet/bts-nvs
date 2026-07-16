"""Unit tests for the Wave-0/Wave-1 SOTA-upgrade components
(Analysis/PLAN_sota_upgrade_2026-07-16.md): the exp037 knapsack encoder, the
exp039 flow-residual alignment, and the exp040 refiner nets/checkpoint format.

These cover the pure logic only -- the parts that do NOT need a nerfstudio
checkpoint load -- so a regression is caught here in seconds instead of on the
GPU fleet.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def _load(mod, rel):
    """Analysis/ filenames start with a digit -> import by path, not by name."""
    spec = importlib.util.spec_from_file_location(mod, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


knap = _load("knap16", "Analysis/16_encode_knapsack.py")
flow = _load("flow17", "Analysis/17_flow_align.py")
ref = _load("ref10", "Analysis/10_refiner_pilot.py")


def _photo(seed=0, size=(96, 64)):
    """Low-res noise upscaled = smooth gradients, so JPEG behaves like a photo."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(8, 12, 3), dtype=np.uint8)
    return Image.fromarray(base).resize(size, Image.BILINEAR)


# ------------------------------- exp037 knapsack -------------------------------
def test_recoverable_score_rises_with_quality():
    im = _photo()
    scores = [knap.recoverable_score(knap._decode(knap.encode(im, q)), im)
              for q in (60, 80, 95)]
    assert scores[0] < scores[1] < scores[2]


def test_allocate_respects_budget_and_prefers_high_quality_when_rich():
    imgs = [(f"{i}.JPG", _photo(seed=i)) for i in range(4)]
    rich = knap.allocate(imgs, budget_bytes=10_000_000, qualities=(80, 90, 96),
                         verbose=False)
    assert set(rich) == {f"{i}.JPG" for i in range(4)}
    # budget is effectively unbounded -> everyone lands on the top rung
    assert all(q == 96 for _, q in rich.values())

    floor = sum(len(knap.encode(im, 80)) for _, im in imgs)
    tight = knap.allocate(imgs, budget_bytes=floor + 10, qualities=(80, 90, 96),
                          verbose=False)
    assert sum(len(d) for d, _ in tight.values()) <= floor + 10


def test_allocate_beats_flat_quality_at_equal_bytes():
    """The knapsack's whole claim: spending the SAME bytes non-uniformly buys at
    least as much summed fidelity as one flat quality for every image."""
    imgs = [(f"{i}.JPG", _photo(seed=i)) for i in range(6)]
    qualities = (80, 86, 92, 96)
    flat_q = 92
    flat_bytes = sum(len(knap.encode(im, flat_q)) for _, im in imgs)
    flat_score = sum(knap.recoverable_score(knap._decode(knap.encode(im, flat_q)), im)
                     for _, im in imgs)

    alloc = knap.allocate(imgs, budget_bytes=flat_bytes, qualities=qualities,
                          verbose=False)
    got_bytes = sum(len(d) for d, _ in alloc.values())
    got_score = sum(knap.recoverable_score(knap._decode(d), dict(imgs)[n])
                    for n, (d, _) in alloc.items())
    assert got_bytes <= flat_bytes
    assert got_score >= flat_score - 1e-6


def test_mozjpeg_detection_rejects_libjpeg_turbo_cjpeg(monkeypatch):
    """libjpeg-turbo also ships `cjpeg` but has no trellis and no -tune-*;
    accepting it blindly makes encode() raise CalledProcessError."""
    class _R:
        stdout, stderr = "libjpeg-turbo version 3.1.4.1", ""
    monkeypatch.setattr(knap.shutil, "which", lambda _: "/usr/bin/cjpeg")
    monkeypatch.setattr(knap.subprocess, "run", lambda *a, **k: _R())
    assert knap._find_mozjpeg() is None

    _R.stdout = "mozjpeg version 4.1.1 (build 20230101)"
    assert knap._find_mozjpeg() == "/usr/bin/cjpeg"


# ------------------------------ exp039 flow align ------------------------------
def _shifted_pair(dx=3, dy=-2, shape=(64, 96)):
    import cv2
    rng = np.random.default_rng(0)
    ref_ = cv2.GaussianBlur(rng.random(shape + (3,)).astype(np.float32), (0, 0), 1.5)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    src = cv2.warpAffine(ref_, M, (shape[1], shape[0]), borderMode=cv2.BORDER_REPLICATE)
    return src, ref_


def test_align_recovers_small_shift():
    src, ref_ = _shifted_pair()
    aligned, mask = flow.align_to_reference(src, ref_, max_px=7, backend="dis")
    assert np.abs(aligned - ref_).mean() < np.abs(src - ref_).mean()
    assert mask.mean() > 0.5


def test_align_never_worsens_a_pixel():
    """Conservative by construction: a pixel keeps the un-aligned value unless
    alignment improved its agreement with the render -- so the guard can only see
    equal-or-better evidence than before."""
    src, ref_ = _shifted_pair()
    aligned, _ = flow.align_to_reference(src, ref_, max_px=7, backend="dis")
    err0 = np.abs(src - ref_).mean(-1)
    err1 = np.abs(aligned - ref_).mean(-1)
    assert (err1 <= err0 + 1e-6).all()


def test_align_output_is_in_range_and_shaped():
    src, ref_ = _shifted_pair()
    aligned, mask = flow.align_to_reference(src, ref_, max_px=7, backend="dis")
    assert aligned.shape == src.shape and mask.shape == src.shape[:2]
    assert aligned.min() >= 0.0 and aligned.max() <= 1.0


def test_align_is_identity_on_identical_inputs():
    _, ref_ = _shifted_pair()
    aligned, mask = flow.align_to_reference(ref_.copy(), ref_, max_px=7, backend="dis")
    assert np.allclose(aligned, ref_, atol=1e-5)
    assert mask.sum() == 0  # nothing to improve -> nothing applied


def test_max_px_clamp_bounds_the_correction():
    """A shift far beyond max_px must not be fully undone -- the clamp is what
    keeps a bad flow estimate from dragging in unrelated texture."""
    src, ref_ = _shifted_pair(dx=20, dy=0)
    aligned, _ = flow.align_to_reference(src, ref_, max_px=2, backend="dis")
    assert np.abs(aligned - ref_).mean() > 0.01


# ------------------------------ exp040 refiner ---------------------------------
@pytest.mark.parametrize("blocks", ["conv", "naf"])
@pytest.mark.parametrize("ci", [7, 16])
def test_unet_forward_shape(blocks, ci):
    net = ref.UNet(ci=ci, co=3, base=8, blocks=blocks)
    out = net(torch.zeros(1, ci, 72, 88))
    assert out.shape == (1, 3, 72, 88)


@pytest.mark.parametrize("blocks", ["conv", "naf"])
def test_checkpoint_roundtrip(tmp_path, blocks):
    net = ref.UNet(ci=16, co=3, base=8, blocks=blocks)
    p = tmp_path / "refiner.pt"
    ref.save_refiner(net, p)
    back = ref.load_refiner(p, "cpu")
    x = torch.randn(1, 16, 32, 40)
    net.eval(); back.eval()
    with torch.no_grad():
        assert torch.allclose(net(x), back(x), atol=1e-6)


def test_load_refiner_accepts_legacy_raw_state_dict(tmp_path):
    """Shipped v1/v2 checkpoints are bare state_dicts with no ci/base/blocks
    metadata; they must keep loading after the v3 format change."""
    net = ref.UNet(ci=7, co=3, base=8, blocks="conv")
    p = tmp_path / "legacy.pt"
    torch.save(net.state_dict(), p)
    back = ref.load_refiner(p, "cpu")
    assert back.ci == 7 and back.base == 8 and back.blocks_kind == "conv"


def test_ensemble_averages_member_residuals():
    """Members share the DIBR base, so the ensemble averages RESIDUALS and clamps
    once -- not the members' already-clamped RGB (which would let one saturated
    member drag the mean)."""
    nets = [ref.UNet(ci=7, co=3, base=8).eval() for _ in range(3)]
    inp = np.random.default_rng(0).random((7, 32, 40)).astype(np.float32)
    got = ref._net_apply_ensemble(nets, inp, "cpu")

    x = torch.from_numpy(inp[None])
    with torch.no_grad():
        res = torch.stack([n(x) for n in nets], 0).mean(0)
        want = (x[:, 3:6] + res).clamp(0, 1)[0].numpy().transpose(1, 2, 0)
    assert got.shape == (32, 40, 3)
    assert np.allclose(got, want, atol=1e-5)


def test_ensemble_matches_member_mean_when_unsaturated():
    """Where no member's output is clipped, residual-averaging and RGB-averaging
    coincide -- pinning the ensemble to the obvious semantics on real inputs
    (refined RGB sits inside [0,1] almost everywhere)."""
    nets = [ref.UNet(ci=7, co=3, base=8).eval() for _ in range(3)]
    for n in nets:  # zero the head -> tanh(0)=0 residual, so nothing saturates
        torch.nn.init.zeros_(n.head.weight); torch.nn.init.zeros_(n.head.bias)
    inp = np.random.default_rng(0).random((7, 32, 40)).astype(np.float32)
    got = ref._net_apply_ensemble(nets, inp, "cpu")
    each = np.stack([ref._net_apply(n, inp, "cpu") for n in nets])
    assert np.allclose(got, each.mean(0), atol=1e-5)
    assert np.allclose(got, inp[3:6].transpose(1, 2, 0), atol=1e-5)
