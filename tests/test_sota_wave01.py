"""Unit tests for the Wave-0/Wave-1 SOTA-upgrade components
(docs/archive_phase1/PLAN_sota_upgrade_2026-07-16.md): the exp037 knapsack encoder, the
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


# -------------------------- exp040 exposure correction -------------------------
def test_exposure_fit_recovers_a_global_gain():
    rng = np.random.default_rng(0)
    ref = rng.random((32, 40, 3)).astype(np.float32) * 0.6 + 0.2
    src = np.clip(ref * 1.15, 0, 1)  # neighbour auto-exposed brighter
    mask = np.ones((32, 40), bool)
    got = dibr._fit_exposure(src, ref, mask)
    assert np.abs(got - ref).mean() < np.abs(src - ref).mean() / 5


def test_exposure_fit_is_skipped_on_thin_evidence():
    """A fit from a handful of pixels is noise; leaving the warp untouched is
    the safer default."""
    rng = np.random.default_rng(0)
    ref = rng.random((32, 40, 3)).astype(np.float32)
    src = rng.random((32, 40, 3)).astype(np.float32)
    mask = np.zeros((32, 40), bool)
    mask[:2, :3] = True
    assert np.allclose(dibr._fit_exposure(src, ref, mask), src)


def test_exposure_fit_ignores_pixels_outside_the_mask():
    """The fit must use only depth-consistent pixels -- occluded junk in the
    warp must not drag the correction."""
    rng = np.random.default_rng(0)
    ref = rng.random((32, 40, 3)).astype(np.float32) * 0.5 + 0.25
    src = np.clip(ref * 1.1, 0, 1)
    mask = np.zeros((32, 40), bool)
    mask[:20] = True
    junk = src.copy()
    junk[20:] = 0.0  # garbage only where mask is False
    a = dibr._fit_exposure(src, ref, mask)[:20]
    b = dibr._fit_exposure(junk, ref, mask)[:20]
    assert np.allclose(a, b, atol=1e-5)


def test_exposure_gain_is_clamped():
    """An extreme gain means the fit is wrong, not that the scene is 8x brighter;
    clamping keeps a bad fit from being worse than no correction."""
    rng = np.random.default_rng(0)
    src = (rng.random((32, 40, 3)).astype(np.float32) * 0.05 + 0.02)
    ref = np.clip(src * 8.0, 0, 1)
    got = dibr._fit_exposure(src, ref, np.ones((32, 40), bool), max_gain=1.25)
    # gain clamped to 1.25 -> cannot reach an 8x-brighter reference
    assert np.abs(got - ref).mean() > 0.05


# --------------------------- exp041 depth import -------------------------------
imp = _load("imp18", "Analysis/18_import_depth.py")


def _depth_field(H=48, W=64):
    """A plausible z-depth map: nearer at the bottom, plus some structure."""
    gy, gx = np.mgrid[0:H, 0:W]
    return (12.0 - 6.0 * gy / H + 0.6 * np.sin(gx / 5.0)).astype(np.float32)


def test_check_depth_accepts_a_matching_export():
    ours = _depth_field()
    theirs = ours + np.random.default_rng(0).normal(0, 0.02, ours.shape).astype(np.float32)
    assert imp.check_depth(ours, theirs) == []


def test_check_depth_flags_scale_mismatch():
    """Their repo renormalising the COLMAP scene is the most likely failure and
    the most silent: every z-test then compares metres to arbitrary units."""
    ours = _depth_field()
    problems = imp.check_depth(ours, ours * 2.5)
    assert any("scale mismatch" in p for p in problems)


def test_check_depth_flags_disparity_export():
    ours = _depth_field()
    problems = imp.check_depth(ours, (1.0 / ours).astype(np.float32))
    assert problems  # inverse depth must never pass silently


def test_check_depth_flags_ray_distance_instead_of_z():
    """Range-from-centre grows off-axis; it looks almost right on-axis, so only
    the radial trend catches it."""
    H, W = 48, 64
    ours = _depth_field(H, W)
    gy, gx = np.mgrid[0:H, 0:W]
    f = 50.0
    theirs = (ours * np.sqrt(1 + ((gx - W / 2) ** 2 + (gy - H / 2) ** 2) / f ** 2)).astype(np.float32)
    problems = imp.check_depth(ours, theirs)
    assert any("RAY DISTANCE" in p for p in problems)


def test_check_depth_flags_shape_and_nan():
    ours = _depth_field()
    assert any("shape" in p for p in imp.check_depth(ours, ours[:, :10]))
    bad = ours.copy(); bad[0, 0] = np.nan
    assert any("NaN" in p for p in imp.check_depth(ours, bad))


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


# --------------------------- exp040 evidence stack -----------------------------
dibr = _load("dibr04", "Analysis/04_x3_dibr_pilot.py")


def _evidence(n_neigh, K=3, H=8, W=10):
    rng = np.random.default_rng(0)
    cols = [rng.random((H, W, 3)).astype(np.float32) for _ in range(n_neigh)]
    confs = [rng.random((H, W)).astype(np.float32) for _ in range(n_neigh)]
    depth = rng.random((H, W)).astype(np.float32) * 10 + 1
    return dibr.Warper._pack_evidence(cols, confs, depth, K, H, W), cols, confs


def test_evidence_layout_and_width():
    ev, cols, confs = _evidence(3)
    assert ev.shape == (8, 10, 4 * 3 + 1)
    for i in range(3):
        assert np.allclose(ev[..., 3 * i:3 * i + 3], cols[i])
        assert np.allclose(ev[..., 9 + i], confs[i])


def test_evidence_pads_missing_neighbours_with_zero_confidence():
    """Fewer than K usable neighbours must not shift the channel meaning --
    slot i is always the i-th nearest, absent slots read as no evidence."""
    ev, cols, _ = _evidence(1)
    assert np.allclose(ev[..., 0:3], cols[0])
    assert np.abs(ev[..., 3:9]).sum() == 0  # empty warp slots
    assert np.abs(ev[..., 10:12]).sum() == 0  # their confidences


def test_evidence_depth_is_normalised_and_bounded():
    ev, _, _ = _evidence(3)
    d = ev[..., -1]
    assert 0.0 <= d.min() and d.max() <= 1.0
    # median depth must land at ~0.5 after the /(2*median) normalisation
    assert abs(float(np.median(d)) - 0.5) < 1e-5


def test_evidence_stack_is_v2_prefix_compatible():
    """v3 keeps [render|DIBR|mask] as the first 7 channels: _net_apply reads the
    residual base from 3:6, so a layout change there would silently corrupt it."""
    rng = np.random.default_rng(1)
    rgb, dib = rng.random((8, 10, 3)).astype(np.float32), rng.random((8, 10, 3)).astype(np.float32)
    mask = rng.random((8, 10)).astype(np.float32)
    ev, _, _ = _evidence(3)
    v2 = ref.stack_channels(rgb, dib, mask)
    v3 = ref.stack_channels(rgb, dib, mask, ev)
    assert v2.shape[-1] == 7 and v3.shape[-1] == 7 + 4 * 3 + 1
    assert np.allclose(v3[..., :7], v2)
    assert np.allclose(v3[..., 3:6].astype(np.float32), dib, atol=1e-3)


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
