"""Fit the effective psnr_max (and set offset) from submission datapoints.

Model: assuming the scorer uses the published formula on the private set and
private-vs-public difficulty is a stable additive offset,
    lb/100 = a + 0.4*(1 - lpips_pub) + 0.3*ssim_pub + (0.3/psnr_max)*psnr_pub
so with y = lb/100 - 0.4*(1-lpips) - 0.3*ssim, a linear fit y = a + b*psnr
gives psnr_max = 0.3/b. Requires >=3 COMPLETE rows with real PSNR spread
(>=0.5 dB) to mean anything -- below that it prints the numbers but tells you
not to act on them.

Escalation rule (plan_overall_v3 §6): implied psnr_max <= 30 means PSNR is
worth more than we assumed -> LPIPS loses primacy, reprioritize.

Input: results/leaderboard_reconciliation.csv (one row per submission; keep
appending). Incomplete rows (missing ssim/lpips) are skipped with a warning.
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def fit(rows: list[dict]) -> dict:
    """rows: dicts with local_psnr, local_ssim, local_lpips, lb_score (floats)."""
    psnr = np.array([r["local_psnr"] for r in rows])
    y = np.array([r["lb_score"] / 100 - 0.4 * (1 - r["local_lpips"]) - 0.3 * r["local_ssim"]
                  for r in rows])
    b, a = np.polyfit(psnr, y, 1)
    pred = a + b * psnr
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": len(rows),
        "psnr_spread": float(psnr.max() - psnr.min()),
        "offset_a": float(a),
        "slope_b": float(b),
        "psnr_max": float(0.3 / b) if b > 0 else float("inf"),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def load_rows(path: Path) -> tuple[list[dict], int]:
    complete, skipped = [], 0
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                complete.append({
                    "local_psnr": float(r["local_psnr"]),
                    "local_ssim": float(r["local_ssim"]),
                    "local_lpips": float(r["local_lpips"]),
                    "lb_score": float(r["lb_score"]),
                })
            except (ValueError, KeyError):
                skipped += 1
    return complete, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("results/leaderboard_reconciliation.csv"))
    args = ap.parse_args()

    rows, skipped = load_rows(args.csv)
    if skipped:
        print(f"skipped {skipped} incomplete row(s) (missing ssim/lpips or lb_score)")
    if len(rows) < 2:
        raise SystemExit(f"only {len(rows)} complete rows -- nothing to fit yet, keep logging.")

    r = fit(rows)
    print(f"n={r['n']}  psnr_spread={r['psnr_spread']:.2f} dB  R^2={r['r2']:.4f}")
    print(f"implied psnr_max = {r['psnr_max']:.1f} dB   set offset a = {r['offset_a']:+.4f}")
    if r["n"] < 3 or r["psnr_spread"] < 0.5:
        print("NOT ACTIONABLE: need >=3 complete points with >=0.5 dB PSNR spread. "
              "Log more submissions before believing this.")
    elif r["psnr_max"] <= 30:
        print("*** ESCALATE (plan_overall_v3 §6): implied psnr_max <= 30 -- PSNR is worth "
              "more than assumed; LPIPS loses primacy, reprioritize Tier A. ***")
    if not (10 <= r["psnr_max"] <= 100):
        print("WARNING: implied psnr_max outside [10,100] -- the additive-offset assumption "
              "is probably violated (metric mismatch or scorer change); investigate before use.")


if __name__ == "__main__":
    main()
