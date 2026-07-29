"""Motion-compensated temporal interpolation pilot for Round-2 indoor scenes.

The indoor scene filenames retain their source-video frame index, and every
test pose is bracketed by two supplied train frames.  This script evaluates a
pretrained arbitrary-timestep RIFE model on the existing honest hold-out:

* hold-out targets come from a baseline ``metrics_val_split.json``;
* by default, every hold-out frame is excluded from every source pair, matching
  the existing 3DGS hold-out's source policy;
* the interpolation time is the exact filename-derived fraction;
* predictions are evaluated with the confirmed VGG/PSNR_MAX=50 grader.

The RIFE implementation and weights stay external to this repository.  A
typical invocation is:

    python Analysis/27_rife_temporal_pilot.py \
      --scene chair \
      --rife-repo /tmp/Practical-RIFE \
      --model-dir /tmp/Practical-RIFE/train_log

For a final GT-less test render, pass ``--mode test``.  Test mode writes images
but deliberately cannot score them.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import importlib
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def frame_id(name: str) -> int:
    match = re.search(r"(\d+)(?=\.[^.]+$)", name)
    if match is None:
        raise ValueError(f"cannot recover a frame index from {name!r}")
    return int(match.group(1))


def holdout_names(metrics_path: Path) -> list[str]:
    data = json.loads(metrics_path.read_text())
    rows = data.get("per_image", [])
    names = [r["image"] for r in rows if r.get("image") != "mean"]
    if not names:
        raise SystemExit(f"{metrics_path}: no per-image hold-out rows")
    return names


def test_names(scene_dir: Path) -> list[str]:
    with open(scene_dir / "test/test_poses.csv") as handle:
        return [row["image_name"] for row in csv.DictReader(handle)]


def source_plan(image_dir: Path, targets: list[str],
                excluded_sources: set[str] | None = None,
                skip_unbracketed: bool = False):
    all_sources = [p.name for p in image_dir.iterdir() if p.is_file()]
    excluded_sources = excluded_sources or set()
    plan = []
    for target in targets:
        sources = sorted((name for name in all_sources
                          if name != target and name not in excluded_sources),
                         key=frame_id)
        source_ids = [frame_id(name) for name in sources]
        target_id = frame_id(target)
        pos = bisect.bisect_left(source_ids, target_id)
        if pos == 0 or pos == len(sources):
            if skip_unbracketed:
                print(f"WARN {target}: not temporally bracketed; omitted from hold-out score")
                continue
            raise SystemExit(f"{target}: not temporally bracketed by source frames")
        lo, hi = sources[pos - 1], sources[pos]
        lo_id, hi_id = frame_id(lo), frame_id(hi)
        alpha = (target_id - lo_id) / (hi_id - lo_id)
        if not 0.0 < alpha < 1.0:
            raise AssertionError((target, lo, hi, alpha))
        plan.append((target, lo, hi, alpha))
    return plan


def load_model(rife_repo: Path, model_dir: Path):
    sys.path.insert(0, str(rife_repo))
    module = importlib.import_module("train_log.RIFE_HDv3")
    model = module.Model()
    model.load_model(str(model_dir), -1)
    model.eval()
    model.device()
    return model


def load_tensor(path: Path, device: str) -> tuple[torch.Tensor, int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"OpenCV could not read {path}")
    height, width = image.shape[:2]
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
    return (tensor.div_(255.0).unsqueeze(0).to(device),
            height, width)


def infer(model, lo_path: Path, hi_path: Path, alpha: float,
          scale: float) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lo, height, width = load_tensor(lo_path, device)
    hi, height_hi, width_hi = load_tensor(hi_path, device)
    if (height_hi, width_hi) != (height, width):
        raise SystemExit(f"source dimensions disagree: {lo_path} vs {hi_path}")
    # This v4.25 model starts at scale 32 and its internal feature encoder adds
    # another factor of four.  A 128-pixel multiple avoids shape disagreement
    # at 1080p (64 is sufficient for some older RIFE variants only).
    pad_h = ((height - 1) // 128 + 1) * 128 - height
    pad_w = ((width - 1) // 128 + 1) * 128 - width
    lo = F.pad(lo, (0, pad_w, 0, pad_h))
    hi = F.pad(hi, (0, pad_w, 0, pad_h))
    with torch.inference_mode():
        pred = model.inference(lo, hi, timestep=float(alpha), scale=scale)
    pred = pred[0, :, :height, :width].clamp(0, 1)
    return (pred.mul(255).byte().cpu().numpy().transpose(1, 2, 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["bonsai", "chair"], required=True)
    parser.add_argument("--mode", choices=["holdout", "test"], default="holdout")
    parser.add_argument("--rife-repo", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="RIFE internal inference scale; use 0.5 only as an OOM fallback")
    parser.add_argument("--metrics-json", type=Path, default=None,
                        help="hold-out membership; defaults to the raw-backbone metrics")
    parser.add_argument(
        "--source-policy", choices=["strict", "leave-one-out"], default="strict",
        help=("strict excludes the entire hold-out from all source pairs, matching "
              "the 3DGS validation policy; leave-one-out excludes only the current "
              "target and is an optimistic approximation to production spacing"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    scene_dir = REPO / "data/raw/phase2" / args.scene
    image_dir = scene_dir / "train/images"
    metrics_path = args.metrics_json or (
        REPO / "runs/round2/val_holdout" / args.scene / "metrics_val_split.json")
    if args.mode == "holdout":
        targets = holdout_names(metrics_path)
    else:
        targets = test_names(scene_dir)

    out_dir = args.out_dir or (
        REPO / "results/rife_temporal_pilot" / args.scene / args.mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    # A match-test hold-out can contain a capture endpoint after its own frame
    # is removed.  Such a row does not represent the actual test set (whose
    # frames are all bracketed), so omit it and compare both arms on the same
    # remaining subset.  Test mode stays strict.
    excluded_sources = (
        set(targets)
        if args.mode == "holdout" and args.source_policy == "strict"
        else set()
    )
    plan = source_plan(
        image_dir, targets, excluded_sources=excluded_sources,
        skip_unbracketed=args.mode == "holdout")
    (out_dir / "source_plan.json").write_text(json.dumps([
        {"target": target, "lo": lo, "hi": hi, "alpha": alpha}
        for target, lo, hi, alpha in plan
    ], indent=2))

    model = load_model(args.rife_repo, args.model_dir)
    for index, (target, lo, hi, alpha) in enumerate(plan, 1):
        pred = infer(model, image_dir / lo, image_dir / hi, alpha, args.scale)
        if not cv2.imwrite(str(out_dir / target), pred,
                           [cv2.IMWRITE_JPEG_QUALITY, 98]):
            raise SystemExit(f"failed to write {out_dir / target}")
        print(f"[{index:02d}/{len(plan)}] {target}: {lo} -> {hi} at t={alpha:.3f}",
              flush=True)

    if args.mode == "holdout":
        from src.metrics import compute_metrics
        metrics = compute_metrics(out_dir, image_dir, "vgg", 50.0)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        mean = metrics["mean"]
        baseline = json.loads(metrics_path.read_text()).get("per_image", [])
        generated = {target for target, _, _, _ in plan}
        baseline = [row for row in baseline if row.get("image") in generated]
        baseline_score = float(np.mean([row["score"] for row in baseline]))
        print(f"RIFE {args.scene}: PSNR={mean['psnr']:.3f} "
              f"SSIM={mean['ssim']:.4f} LPIPS={mean['lpips']:.4f} "
              f"Score={mean['score']:.4f} (n={len(plan)}); "
              f"raw-backbone same-subset Score={baseline_score:.4f}")
    else:
        print(f"Wrote {len(plan)} GT-less test predictions to {out_dir}")


if __name__ == "__main__":
    main()
