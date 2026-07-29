"""PairPool must be a drop-in for the old in-RAM loader (2026-07-19).

The old loader inflated every pair to float32 and kept it resident, capping the
20ch evidence stack at 60 pairs on an 11 GB box. PairPool memmaps per-pair .npy
sidecars and slices only the training crop. That is only a safe swap if the
crops it returns are IDENTICAL to what the old loader produced -- otherwise
every refiner number measured before and after is on a different footing and the
0.7710 regression gate is meaningless.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "refiner10", REPO / "Analysis" / "10_refiner_pilot.py")
refiner10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refiner10)
PairPool = refiner10.PairPool


def _make_cache(tmp_path, n=3, H=40, W=52, ci=7):
    """Write savez_compressed pairs shaped like the real cache (f16 HWC)."""
    rng = np.random.default_rng(0)
    names = []
    for i in range(n):
        nm = f"frame_{i:03d}"
        np.savez_compressed(
            tmp_path / f"{nm}.npz",
            inp=rng.random((H, W, ci)).astype(np.float16),
            tgt=rng.random((H, W, 3)).astype(np.float16))
        names.append(nm)
    return names


def _old_loader_crop(cache, nm, y0, x0, c):
    """Exactly what the pre-memmap loader did: whole-array f32 upcast, then slice."""
    d = np.load(cache / f"{nm}.npz")
    inp, tgt = d["inp"].astype(np.float32), d["tgt"].astype(np.float32)
    return inp[y0:y0 + c, x0:x0 + c], tgt[y0:y0 + c, x0:x0 + c]


def test_crops_match_old_loader_exactly(tmp_path):
    names = _make_cache(tmp_path)
    pool = PairPool(tmp_path, names, verbose=False)
    for i, nm in enumerate(names):
        for (y0, x0, c) in [(0, 0, 16), (7, 11, 16), (24, 36, 16)]:
            got_i, got_t = pool.crop(i, y0, x0, c)
            exp_i, exp_t = _old_loader_crop(tmp_path, nm, y0, x0, c)
            # bit-exact: f16->f32 is lossless and slicing is exact
            assert np.array_equal(got_i, exp_i)
            assert np.array_equal(got_t, exp_t)
            assert got_i.dtype == np.float32 and got_t.dtype == np.float32


def test_reports_shape_and_channels(tmp_path):
    names = _make_cache(tmp_path, n=2, H=40, W=52, ci=20)
    pool = PairPool(tmp_path, names, verbose=False)
    assert len(pool) == 2
    assert pool.shape(0) == (40, 52)      # from tgt, as sample_batch expects
    assert pool.ci == 20                  # evidence stack width, not a constant


def test_transcode_is_idempotent_and_cached(tmp_path):
    """Second construction must reuse sidecars, not rewrite them."""
    names = _make_cache(tmp_path, n=2)
    PairPool(tmp_path, names, verbose=False)
    mm = tmp_path / "_mm"
    stamps = {p.name: p.stat().st_mtime_ns for p in mm.iterdir()}
    assert len(stamps) == 4               # inp+tgt per pair
    PairPool(tmp_path, names, verbose=False)
    assert {p.name: p.stat().st_mtime_ns for p in mm.iterdir()} == stamps


def test_pool_is_lazy_not_resident(tmp_path):
    """The point of the class: arrays stay memmaps, so RAM is O(1) in pairs."""
    names = _make_cache(tmp_path, n=3)
    pool = PairPool(tmp_path, names, verbose=False)
    assert all(isinstance(a, np.memmap) for a in pool._inp)
    assert all(isinstance(a, np.memmap) for a in pool._tgt)


def test_subset_pools_share_one_sidecar_dir(tmp_path):
    """train/val pools are built from disjoint name lists over the same cache;
    each must transcode only its own pairs and not trip over the other's."""
    names = _make_cache(tmp_path, n=4)
    fit = PairPool(tmp_path, names[:3], verbose=False)
    val = PairPool(tmp_path, names[3:], verbose=False)
    assert len(fit) == 3 and len(val) == 1
    assert len(list((tmp_path / "_mm").iterdir())) == 8


@pytest.mark.parametrize("ci", [7, 20])
def test_channel_widths_roundtrip(tmp_path, ci):
    names = _make_cache(tmp_path, n=1, ci=ci)
    pool = PairPool(tmp_path, names, verbose=False)
    inp, _ = pool.crop(0, 0, 0, 8)
    assert inp.shape == (8, 8, ci)


def test_full_pair_matches_old_loader_exactly(tmp_path):
    names = _make_cache(tmp_path, n=1, H=40, W=52, ci=20)
    pool = PairPool(tmp_path, names, verbose=False)
    got_i, got_t = pool.full(0)
    d = np.load(tmp_path / f"{names[0]}.npz")
    assert np.array_equal(got_i, d["inp"].astype(np.float32))
    assert np.array_equal(got_t, d["tgt"].astype(np.float32))
