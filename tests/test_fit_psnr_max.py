import numpy as np

from src.fit_psnr_max import fit


def _synth_rows(psnr_max, offset, psnrs, ssim=0.71, lpips=0.16):
    rows = []
    for p in psnrs:
        lb = 100 * (offset + 0.4 * (1 - lpips) + 0.3 * ssim + 0.3 * p / psnr_max)
        rows.append({"local_psnr": p, "local_ssim": ssim, "local_lpips": lpips, "lb_score": lb})
    return rows


def test_recovers_known_psnr_max():
    rows = _synth_rows(psnr_max=40.0, offset=-0.135, psnrs=[20.5, 21.2, 22.0, 23.1])
    r = fit(rows)
    assert abs(r["psnr_max"] - 40.0) < 0.01
    assert abs(r["offset_a"] - (-0.135)) < 1e-6
    assert r["r2"] > 0.999


def test_recovers_with_varying_other_metrics():
    # ssim/lpips vary across submissions too -- fit must still recover psnr_max
    rows = []
    for p, s, l in [(20.5, 0.70, 0.17), (21.2, 0.71, 0.16), (22.0, 0.72, 0.15), (23.0, 0.735, 0.14)]:
        lb = 100 * (-0.135 + 0.4 * (1 - l) + 0.3 * s + 0.3 * p / 35.0)
        rows.append({"local_psnr": p, "local_ssim": s, "local_lpips": l, "lb_score": lb})
    r = fit(rows)
    assert abs(r["psnr_max"] - 35.0) < 0.01


def test_noise_tolerance():
    rng = np.random.default_rng(0)
    rows = _synth_rows(40.0, -0.135, psnrs=[20.0, 21.0, 22.0, 23.0, 24.0])
    for row in rows:
        row["lb_score"] += rng.normal(0, 0.02)  # +-0.0002 on /100 scale
    r = fit(rows)
    assert 35 < r["psnr_max"] < 46
