"""Matched full-frame DIBR source-policy benchmark on supplied training views.

The selected frames are removed from the renderer's source pool and from the
3DGS training set.  This measures a source-policy change without test GT or
test-specific tuning.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


dibr = load_module("dibr04", REPO / "Analysis/04_x3_dibr_pilot.py")
metrics = load_module("competition_metrics", REPO / "src/metrics.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="bonsai")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--policies", nargs="+", default=["spatial", "temporal"])
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--guard", type=float, default=0.18)
    ap.add_argument("--ss", type=int, default=1)
    ap.add_argument("--sample", choices=["bilinear", "cubic"], default="bilinear")
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "results/source_policy_holdout",
    )
    args = ap.parse_args()

    holdout = [line.strip() for line in args.split.read_text().splitlines() if line.strip()]
    if not holdout:
        raise SystemExit(f"empty split: {args.split}")

    warper = dibr.Warper(
        args.scene,
        config_path=args.config,
        ss=args.ss,
        sample=args.sample,
        holdout_names=holdout,
    )
    if set(holdout) != {name for name, _ in warper.holdout_poses}:
        raise RuntimeError("holdout/source exclusion mismatch")

    gt_dir = dibr.scene_raw(args.scene) / args.scene / "train/images"
    result = {
        "scene": args.scene,
        "config": str(args.config),
        "holdout": holdout,
        "K": args.K,
        "tol": args.tol,
        "guard": args.guard,
        "ss": args.ss,
        "sample": args.sample,
        "policies": {},
    }
    for policy in args.policies:
        render_dir = args.out / args.scene / policy / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        fallback, guard_reject = [], []
        for n, (name, c2w) in enumerate(warper.holdout_poses, 1):
            out, fallback_frac, _ = warper.synthesize(
                c2w,
                warper.f,
                warper.f,
                warper.cx,
                warper.cy,
                warper.W_tr,
                warper.H_tr,
                K=args.K,
                exclude_names={name},
                tol=args.tol,
                out_k=warper.k,
                guard=args.guard,
                override_name=name,
                source_policy=policy,
            )
            Image.fromarray(np.rint(out * 255).astype(np.uint8)).save(
                render_dir / name, quality=98, subsampling=0
            )
            fallback.append(float(fallback_frac))
            guard_reject.append(float(warper.last_stats["guard_reject_frac"]))
            print(
                f"{policy:8s} {n:2d}/{len(warper.holdout_poses)} {name} "
                f"fallback={fallback_frac:.3f}",
                flush=True,
            )

        scored = metrics.compute_metrics(render_dir, gt_dir, "vgg", 50.0)
        policy_result = {
            "mean": scored["mean"],
            "fallback_mean": float(np.mean(fallback)),
            "guard_reject_mean": float(np.mean(guard_reject)),
            "per_image": scored["per_image"],
        }
        result["policies"][policy] = policy_result
        mean = scored["mean"]
        print(
            f"{policy}: Score={mean['score']:.6f} PSNR={mean['psnr']:.4f} "
            f"SSIM={mean['ssim']:.6f} LPIPS={mean['lpips']:.6f}",
            flush=True,
        )

    baseline = result["policies"][args.policies[0]]["mean"]
    result["deltas_vs_first"] = {}
    for policy in args.policies[1:]:
        mean = result["policies"][policy]["mean"]
        result["deltas_vs_first"][policy] = {
            key: float(mean[key] - baseline[key])
            for key in ("score", "psnr", "ssim", "lpips")
        }
    out_json = args.out / args.scene / "metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
