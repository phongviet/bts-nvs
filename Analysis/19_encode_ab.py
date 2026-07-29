"""Analysis 19 (exp037 validation): does the knapsack allocation actually raise
the REAL grader Score at equal bytes, or only its own GT-free proxy?

The knapsack (16_encode_knapsack.py) allocates bytes by a proxy — decoded-vs-
lossless SSIM+PSNR — because private scenes have no GT. That proxy is an
assumption, and the unit test only proves the allocation beats a flat quality
ON THE PROXY. This script closes the loop on the one scene where we can: encode
the same lossless renders both ways at EQUAL BYTES and score both against real
test GT with the actual grader (LPIPS-vgg, psnr_max=50).

Honest limits of this measurement:
  * ONE scene (hcm0034) — the only public scene with lossless PNG renders on
    disk; the rest are already JPEG, so re-encoding them would double-encode and
    confound the comparison (E1 measured double-encode at ~-0.001).
  * hcm0034 is the QUIET pilot. The knapsack's whole thesis is spending bytes
    where images differ in JPEG cost, so a single quiet scene is close to its
    WORST case and understates the cross-scene gain on the real 434-image
    private set (sky-heavy vs busy frames). Read the result as a floor and a
    sign check, not as the expected private delta.

Run: conda run -n airace python Analysis/19_encode_ab.py --scene hcm0034
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load(mod, rel):
    spec = importlib.util.spec_from_file_location(mod, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


knap = _load("knap16", "Analysis/16_encode_knapsack.py")


def _write(enc, d: Path):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for name, data in enc.items():
        (d / name).write_bytes(data)
    return sum(len(v) for v in enc.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="hcm0034")
    ap.add_argument("--renders", default=None,
                    help="dir of LOSSLESS png renders (default X5_refiner/{scene}/renders_refined_v2)")
    ap.add_argument("--flat-q", type=int, default=95, help="the shipped flat quality to beat")
    ap.add_argument("--qmin", type=int, default=88)
    ap.add_argument("--qmax", type=int, default=98)
    ap.add_argument("--qstep", type=int, default=2)
    ap.add_argument("--budget-scale", type=float, default=1.0,
                    help="knapsack budget as a multiple of the flat arm's bytes. "
                         "1.0 = equal-byte (tests ALLOCATION). >1 emulates the "
                         "headroom a flat quality leaves unspent because it can "
                         "only step in whole q units (exp034 shipped 321.7 MB of "
                         "a 340 MB cap: q96 globally would have been 353 MB).")
    ap.add_argument("--work", default="/tmp/exp037_ab")
    args = ap.parse_args()

    from src.metrics import compute_metrics

    rd = Path(args.renders) if args.renders else \
        REPO / f"Analysis/X5_refiner/{args.scene}/renders_refined_v2"
    pngs = sorted(rd.glob("*.png"))
    if not pngs:
        raise SystemExit(f"no lossless *.png under {rd} — this A/B needs unencoded "
                         f"renders, or the comparison measures double-encoding")
    # A render dir can hold BOTH the lossless png and an already-encoded jpg of
    # the same frame (load_dir maps a png stem to {stem}.JPG, so the two collide
    # on name). Take the png only — sourcing any arm from the jpg would silently
    # double-encode that frame and wreck the comparison.
    from PIL import Image as _Image
    imgs = [(f.stem + ".JPG", _Image.open(f).convert("RGB")) for f in pngs]
    assert len({n for n, _ in imgs}) == len(imgs), "duplicate render names"
    print(f"{args.scene}: {len(imgs)} lossless png renders from {rd.name} "
          f"({len(list(rd.glob('*.JPG')))} jpg siblings ignored)")

    gt = REPO / f"data/raw/phase1/public_set/{args.scene}/test/images"
    if not gt.exists():
        raise SystemExit(f"no test GT at {gt} — A/B needs a public scene")

    work = Path(args.work)
    # --- arm A: flat quality (what exp034 shipped) ---
    flat = {n: knap.encode(im, args.flat_q, subsampling=0) for n, im in imgs}
    b_flat = _write(flat, work / "flat")
    m_flat = compute_metrics(work / "flat", gt, "vgg", 50.0)["mean"]

    # --- arm B: knapsack at the SAME byte budget ---
    budget = int(b_flat * args.budget_scale)
    alloc = knap.allocate(imgs, budget,
                          qualities=range(args.qmin, args.qmax + 1, args.qstep),
                          subsampling=0, tune="ms-ssim")
    b_knap = _write({n: d for n, (d, _) in alloc.items()}, work / "knap")
    m_knap = compute_metrics(work / "knap", gt, "vgg", 50.0)["mean"]

    qh = {}
    for _, q in alloc.values():
        qh[q] = qh.get(q, 0) + 1
    print(f"\n--- exp037 A/B on {args.scene} ({len(imgs)} images, real grader) ---")
    print(f"flat q{args.flat_q}: {b_flat/1e6:6.2f} MB  Score={m_flat['score']:.5f}  "
          f"PSNR={m_flat['psnr']:.3f} SSIM={m_flat['ssim']:.4f} LPIPS={m_flat['lpips']:.4f}")
    print(f"knapsack   : {b_knap/1e6:6.2f} MB  Score={m_knap['score']:.5f}  "
          f"PSNR={m_knap['psnr']:.3f} SSIM={m_knap['ssim']:.4f} LPIPS={m_knap['lpips']:.4f}")
    print(f"quality histogram: {dict(sorted(qh.items()))}  "
          f"backend={'mozjpeg' if knap.MOZJPEG else 'pillow'}")
    d = m_knap["score"] - m_flat["score"]
    print(f"\nDELTA Score = {d:+.5f} ({d*100:+.3f} LB pts) at "
          f"{(b_knap-b_flat)/1e6:+.2f} MB")
    if b_knap > b_flat:
        print("  !! knapsack used MORE bytes — comparison is not equal-byte, ignore the delta")
    print("  NOTE: one QUIET scene = near-worst case for cross-image allocation; "
          "treat as a floor, not the private-set estimate.")


if __name__ == "__main__":
    main()
