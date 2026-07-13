"""Analysis 15: assemble the exp034 ladder results into one table.

Collects every measured rung (DIBR-level arms from X3_dibr metrics jsons,
refiner-level runs from X5_refiner metrics jsons) for the pilot scenes and
prints the ladder + the public-5 projection versus the top-1 line (0.7702).

Run: conda run -n airace python Analysis/15_ladder_readout.py
"""
from __future__ import annotations

import json
from pathlib import Path

A = Path(__file__).resolve().parent
X3, X5 = A / "X3_dibr", A / "X5_refiner"
TOP1 = 0.7702430

PILOTS = ["hcm0034", "HCM0181"]
DIBR_TAGS = [("baseline (bilinear 1x)", "_g0.18"), ("cubic", "_g0.18_cub"),
             ("ss2", "_g0.18_ss2"), ("ss2+cubic", "_g0.18_ss2cub"),
             ("ss2+cubic+big", "_g0.18_ss2cubbig")]
REF_TAGS = [("refiner v1", ""), ("v1 + TTA", "_tta"), ("v2 (v1 inputs)", "_v2"),
            ("v2 + ss2cub", "_v2sc"), ("v2 + ss2cub + big", "_v2scb")]


def m(path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    d = d.get("mean", d) or {}
    return d if "score" in d else None


def row(name, d):
    if d is None:
        return f"  {name:24s} —"
    return (f"  {name:24s} Score={d['score']:.4f} PSNR={d['psnr']:.3f} "
            f"SSIM={d['ssim']:.4f} LPIPS={d['lpips']:.4f}")


def main():
    for s in PILOTS:
        print(f"== {s} — DIBR level ==")
        for name, tag in DIBR_TAGS:
            print(row(name, m(X3 / s / f"metrics{tag}.json")))
        print(f"== {s} — refiner level ==")
        for name, tag in REF_TAGS:
            print(row(name, m(X5 / s / f"metrics_refined{tag}.json")))
        print()

    # public-5 best-known refiner scores (v2-stack where present, else v1)
    print("== public-5 projection (float level; encode ≈ -0.0011, single-encode q95 4:4:4) ==")
    tot, used = 0.0, []
    for s in ["hcm0031", "hcm0034", "HCM0181", "HCM0193", "HCM0204"]:
        best, src = None, "?"
        for name, tag in reversed(REF_TAGS):
            d = m(X5 / s / f"metrics_refined{tag}.json")
            if d is not None:
                best, src = d["score"], name
                break
        tot += best
        used.append(f"{s}={best:.4f}({src})")
    mean = tot / 5
    print("  " + "  ".join(used))
    print(f"  mean={mean:.4f}  vs top-1 line {TOP1:.4f}  margin={mean - TOP1:+.4f}")


if __name__ == "__main__":
    main()
