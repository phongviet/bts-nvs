"""A.6a test-pose coverage diagnostic + Phase-2/3 regime tripwire.

Per test pose (from test/test_poses.csv), against the leak-filtered train
poses (colmap_train_only/images.bin): nearest-train-pose distance (absolute
and as a fraction of scene extent), angular gap at/near it, and the count of
train views that are "near" (close in position AND viewing angle).

Outputs (appends one block per scene; safe to rerun -- rewrites that scene's rows):
  results/test_pose_coverage.csv          per-pose rows
  results/test_pose_coverage_summary.csv  per-scene percentiles + regime verdict
  results/plots/coverage_<scene>.png      histograms (dist_frac, angle, n_near)

Regime verdict heuristic (tunable via CLI): a scene is flagged EXTRAPOLATIVE if
  p90 min_angle > --extrap-angle (default 20 deg; test view DIRECTIONS unseen
      anywhere in the train set), or
  p90 nearest_dist_frac > --extrap-dist-frac (default 0.15; test POSITIONS far
      from every train camera), or
  frac_uncovered > --extrap-uncovered (default 0.25; fraction of test poses
      with zero train views inside the LOOSE near window: dist_frac<0.10 &
      angle<40 deg).
Calibrated on Phase-1 (2026-07-07): all 13 scenes come out interpolative with
wide margin (min_angle_p90 <= 8.6, dist_frac_p90 <= 0.076, frac_uncovered = 0);
angle_at_nearest alone is misleading (nearest-position cam often looks
elsewhere while another nearby cam matches). In Phase 2/3 this is the hour-0
check that decides whether the enhancer gate loosens fleet-wide.

Usage:
  python src/data_prep/test_pose_coverage.py \
      --raw-root data/raw/phase1 --processed-root data/processed/phase1 \
      --out-csv results/test_pose_coverage.csv [--scenes hcm0034 HCM0181]
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.pose_utils import coverage_stats, load_poses_from_colmap, load_poses_from_csv  # noqa: E402

PER_POSE_FIELDS = ["scene", "image_name", "nearest_dist", "nearest_dist_frac",
                   "nearest_train", "angle_at_nearest", "min_angle", "n_near", "n_near_loose"]
SUMMARY_FIELDS = ["scene", "n_test", "n_train",
                  "dist_frac_p50", "dist_frac_p90", "dist_frac_max",
                  "angle_p50", "angle_p90", "angle_max",
                  "min_angle_p90", "frac_uncovered",
                  "n_near_min", "n_near_p10", "regime"]


def find_scene_dirs(raw_root: Path, scenes: list[str] | None) -> list[Path]:
    dirs = []
    for split in sorted(raw_root.iterdir()):
        if not split.is_dir() or split.name.startswith("__"):
            continue
        for scene_dir in sorted(split.iterdir()):
            if (scene_dir / "test" / "test_poses.csv").exists():
                if scenes is None or scene_dir.name in scenes:
                    dirs.append(scene_dir)
    return dirs


def analyze_scene(scene_dir: Path, processed_root: Path,
                  extrap_dist_frac: float, extrap_angle: float,
                  extrap_uncovered: float):
    scene = scene_dir.name
    train_sparse = processed_root / scene / "colmap_train_only"
    if not (train_sparse / "images.bin").exists():
        # fall back to raw sparse (contains test poses too -- coverage then optimistic)
        train_sparse = scene_dir / "train" / "sparse" / "0"
        print(f"WARN {scene}: colmap_train_only missing, using raw sparse "
              f"(includes test poses -- coverage will look better than reality)")
    train = load_poses_from_colmap(train_sparse)
    test = load_poses_from_csv(scene_dir / "test" / "test_poses.csv")
    rows = coverage_stats(test, train)
    for r in rows:
        r["scene"] = scene

    dist_frac = np.array([r["nearest_dist_frac"] for r in rows])
    angle = np.array([r["angle_at_nearest"] for r in rows])
    min_angle = np.array([r["min_angle"] for r in rows])
    n_near = np.array([r["n_near"] for r in rows])
    n_near_loose = np.array([r["n_near_loose"] for r in rows])
    frac_uncovered = float((n_near_loose == 0).mean())
    regime = ("extrapolative"
              if (np.percentile(min_angle, 90) > extrap_angle
                  or np.percentile(dist_frac, 90) > extrap_dist_frac
                  or frac_uncovered > extrap_uncovered)
              else "interpolative")
    summary = {
        "scene": scene, "n_test": len(test), "n_train": len(train),
        "dist_frac_p50": round(float(np.percentile(dist_frac, 50)), 5),
        "dist_frac_p90": round(float(np.percentile(dist_frac, 90)), 5),
        "dist_frac_max": round(float(dist_frac.max()), 5),
        "angle_p50": round(float(np.percentile(angle, 50)), 2),
        "angle_p90": round(float(np.percentile(angle, 90)), 2),
        "angle_max": round(float(angle.max()), 2),
        "min_angle_p90": round(float(np.percentile(min_angle, 90)), 2),
        "frac_uncovered": round(frac_uncovered, 3),
        "n_near_min": int(n_near.min()),
        "n_near_p10": int(np.percentile(n_near, 10)),
        "regime": regime,
    }
    return rows, summary


def plot_scene(rows: list[dict], scene: str, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, key, title in zip(
            axes,
            ["nearest_dist_frac", "angle_at_nearest", "n_near"],
            ["nearest train dist / scene extent", "angle at nearest (deg)", "# near train views"]):
        ax.hist([r[key] for r in rows], bins=30, color="steelblue")
        ax.set_title(f"{scene}: {title}", fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def upsert_csv(path: Path, fields: list[str], new_rows: list[dict], key_scene: str):
    """Rewrite the CSV keeping other scenes' rows, replacing key_scene's."""
    existing = []
    if path.exists():
        with open(path) as f:
            existing = [r for r in csv.DictReader(f) if r["scene"] != key_scene]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in existing + new_rows:
            w.writerow({k: r[k] for k in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw/phase1"))
    ap.add_argument("--processed-root", type=Path, default=Path("data/processed/phase1"))
    ap.add_argument("--out-csv", type=Path, default=Path("results/test_pose_coverage.csv"))
    ap.add_argument("--plots-dir", type=Path, default=Path("results/plots"))
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--extrap-dist-frac", type=float, default=0.15)
    ap.add_argument("--extrap-angle", type=float, default=20.0)
    ap.add_argument("--extrap-uncovered", type=float, default=0.25)
    args = ap.parse_args()

    scene_dirs = find_scene_dirs(args.raw_root, args.scenes)
    if not scene_dirs:
        raise SystemExit(f"No scenes with test_poses.csv found under {args.raw_root}")

    summary_csv = args.out_csv.with_name(args.out_csv.stem + "_summary.csv")
    for scene_dir in scene_dirs:
        rows, summary = analyze_scene(scene_dir, args.processed_root,
                                      args.extrap_dist_frac, args.extrap_angle,
                                      args.extrap_uncovered)
        upsert_csv(args.out_csv, PER_POSE_FIELDS, rows, scene_dir.name)
        upsert_csv(summary_csv, SUMMARY_FIELDS, [summary], scene_dir.name)
        plot_scene(rows, scene_dir.name, args.plots_dir / f"coverage_{scene_dir.name}.png")
        print(f"{summary['scene']}: regime={summary['regime']} "
              f"dist_frac_p90={summary['dist_frac_p90']} min_angle_p90={summary['min_angle_p90']} "
              f"frac_uncovered={summary['frac_uncovered']} n_near_min={summary['n_near_min']}")
    print(f"Wrote {args.out_csv}, {summary_csv}, plots in {args.plots_dir}/")


if __name__ == "__main__":
    main()
