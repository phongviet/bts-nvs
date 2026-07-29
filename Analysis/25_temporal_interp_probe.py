"""Analysis 25: is the indoor regime actually a VIDEO INTERPOLATION problem?

Every round-2 indoor test pose is temporally *bracketed* by train frames on the
original video's frame grid (bonsai: +-10 frames, chair: +-5). That is a very
different situation from the drone scenes, where test poses are genuine novel
viewpoints. If the two temporal neighbours already carry the right exposure,
the right motion blur and the right sensor noise, then a trivial 2D combination
of them may beat a full 3DGS + DIBR + refiner stack on LPIPS -- the metric that
carries 0.4 of the grader weight and where indoor loses.

This probe scores three zero-parameter temporal baselines on the SAME 25-frame
match-test hold-out used by every other indoor A/B, so the numbers drop straight
into the panel next to the shipped refiner (bonsai 0.6913, chair 0.6650).

  T1 nearest  : copy the temporally nearest remaining train frame
  T2 mean     : 50/50 average of the bracketing train frames
  T3 wmean    : frame-distance-weighted average of the bracketing frames

Sources exclude the hold-out frames, exactly like the Warper does, so nothing
leaks. No model, no training, no GPU beyond the LPIPS forward pass.

Run: conda run -n airace python Analysis/25_temporal_interp_probe.py --scene bonsai
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "Analysis" / "X5_refiner"


def frame_id(name: str) -> int:
    return int(re.findall(r"\d+", name)[0])


def read_holdout(scene: str, phase: str = "round2") -> list[str]:
    f = REPO / "data" / "processed" / phase / scene / "splits" / "val_ids.txt"
    if not f.exists():
        raise SystemExit(f"{scene}: no val split at {f}")
    return sorted(f.read_text().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="bonsai")
    ap.add_argument("--phase", default="round2")
    ap.add_argument("--quality", type=int, default=98,
                    help="JPEG quality, matched to the refiner's apply_val")
    args = ap.parse_args()
    scene = args.scene

    img_dir = REPO / "data" / "raw" / "VAI_NVS_DATA_ROUND2" / scene / "train" / "images"
    holdout = read_holdout(scene, args.phase)
    hold_set = set(holdout)
    # sources = every train frame the Warper would be allowed to see
    srcs = sorted((p.name for p in img_dir.iterdir() if p.name not in hold_set),
                  key=frame_id)
    src_ids = [frame_id(n) for n in srcs]

    print(f"{scene}: {len(srcs)} source frames, {len(holdout)} hold-out frames")

    arms = {"t1_nearest": {}, "t2_mean": {}, "t3_wmean": {}}
    gaps = []
    for name in holdout:
        t = frame_id(name)
        i = bisect.bisect_left(src_ids, t)
        lo = srcs[i - 1] if i > 0 else None
        hi = srcs[i] if i < len(srcs) else None
        dlo = t - frame_id(lo) if lo else 10 ** 6
        dhi = frame_id(hi) - t if hi else 10 ** 6
        gaps.append(min(dlo, dhi))

        near = lo if dlo <= dhi else hi
        arms["t1_nearest"][name] = [(near, 1.0)]
        if lo is None or hi is None:
            arms["t2_mean"][name] = [(near, 1.0)]
            arms["t3_wmean"][name] = [(near, 1.0)]
        else:
            arms["t2_mean"][name] = [(lo, 0.5), (hi, 0.5)]
            w = dhi / (dlo + dhi)  # closer frame gets more weight
            arms["t3_wmean"][name] = [(lo, w), (hi, 1.0 - w)]

    g = np.array(gaps)
    print(f"  frame distance to nearest source: med {np.median(g):.0f} "
          f"p90 {np.percentile(g, 90):.0f} max {g.max():.0f}")

    from src.metrics import compute_metrics

    gt = OUT / scene / "gt_val"
    if not gt.is_dir():
        raise SystemExit(f"{gt} missing -- run the refiner --val-holdout first")

    results = {}
    for arm, plan in arms.items():
        rdir = OUT / scene / f"renders_val_{arm}"
        rdir.mkdir(parents=True, exist_ok=True)
        for name, mix in plan.items():
            acc = None
            for src, wt in mix:
                a = np.asarray(Image.open(img_dir / src).convert("RGB"), np.float32)
                acc = a * wt if acc is None else acc + a * wt
            img = np.clip(acc, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(rdir / name, quality=args.quality)
        m = compute_metrics(rdir, gt, "vgg", 50.0)["mean"]
        results[arm] = m
        print(f"  {arm:12s} Score {m['score']:.4f}  PSNR {m['psnr']:.3f}  "
              f"SSIM {m['ssim']:.4f}  LPIPS {m['lpips']:.4f}")

    (OUT / scene / "metrics_val_temporal.json").write_text(json.dumps(results, indent=2))

    ref = {"bonsai": 0.6913, "chair": 0.6650}.get(scene)
    if ref:
        best = max(results.values(), key=lambda m: m["score"])
        print(f"\n  shipped refiner reference: {ref:.4f}")
        print(f"  best temporal baseline   : {best['score']:.4f} "
              f"({best['score'] - ref:+.4f})")


if __name__ == "__main__":
    main()
