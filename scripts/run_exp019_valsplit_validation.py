"""exp019 -- validate the match-test val split's *selection signal* on public
scenes before trusting it to model-select the 8 private scenes.

Question: does a match-test val split (make_val_split.py --mode match-test,
A.6b) rank config variants more consistently with the real-test-GT ranking
than the old every-Nth split? Public scenes are the only place we can answer
it, because only there do we have both train photos (val GT) and real test GT.

Method, per pilot scene (hcm0034, HCM0181):
  1. Cut BOTH splits at equal size (n_val=30): every-Nth and match-test,
     written to data/processed/phase1/<scene>/splits_exp019/{nth,match}/.
  2. For every existing trained variant of that scene (exp001 sparse, exp002
     dense, exp004 backend variants), render the union of both val sets from
     the existing checkpoint (local GPU, no training).
  3. Score renders vs the real train photos; aggregate per split; read the
     variant's real-test-GT score from its metrics_val.json.
  4. Report, per scene x split: Spearman rho vs the test-GT ranking and the
     pairwise A/B decision-agreement rate.

Decision rule (logged to results/PROGRESS.md): use whichever split ranks
better for private-scene selection; if match-test ranks worse/noisily, do NOT
roll it out on Day-1 faith.

CAVEAT (scope): these checkpoints were trained on ALL train images, so val
views are seen-in-training. This validates the pose-distribution component of
the selection signal only, not holdout generalization -- but the pose
distribution is exactly what A.6b changes.

Usage (airace env):
  python scripts/run_exp019_valsplit_validation.py [--dry-run] [--include-vggt]
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# exp004's mcmc checkpoint uses the custom splatfacto-mcmc method; nerfstudio's
# registry reads this env var at first import, so set it before eval_setup.
from scripts.run_sweep import METHOD_CONFIGS_ENV  # noqa: E402
os.environ.setdefault("NERFSTUDIO_METHOD_CONFIGS", METHOD_CONFIGS_ENV)

SCENES = ["hcm0034", "HCM0181"]
N_VAL = 30
RAW_ROOT = REPO / "data/raw/phase1/public_set"
PROC_ROOT = REPO / "data/processed/phase1"
RUNS_ROOT = REPO / "runs/phase1"
RESULTS_CSV = REPO / "results/exp019_valsplit_validation.csv"
SUMMARY_CSV = REPO / "results/exp019_valsplit_summary.csv"
# A/B deltas smaller than this on test GT are ties we don't grade agreement on
# (adopted-lever scale: antialiased was +0.0016 mean).
MIN_TEST_DELTA = 0.001


# ---------------------------------------------------------------- discovery

def discover_variants(scene: str, include_vggt: bool) -> list[tuple[str, Path]]:
    """(variant_name, run_dir) for every trained checkpoint of this scene."""
    cands = [
        ("exp001_sparse", RUNS_ROOT / "exp001_baseline_splatfacto" / scene),
        ("exp002_dense", RUNS_ROOT / "exp002_dense_colmap_init" / scene),
    ]
    if include_vggt:
        cands.append(("exp003_vggt", RUNS_ROOT / "exp003_vggt_init" / scene))
    for v in ("antialiased", "mcmc", "scale_reg", "sky_mask"):
        # hcm0034's exp004 dirs are bare-named; other scenes are prefixed
        for d in (RUNS_ROOT / "exp004_backend_ablation" / f"{scene}_{v}",
                  RUNS_ROOT / "exp004_backend_ablation" / v if scene == "hcm0034" else None):
            if d is not None and d.is_dir():
                cands.append((f"exp004_{v}", d))
                break
    return [(n, d) for n, d in cands if d.is_dir() and find_config(d) is not None]


def find_config(run_dir: Path) -> Path | None:
    hits = sorted(run_dir.glob("train_staging*/splatfacto*/*/config.yml"))
    return hits[-1] if hits else None


def test_gt_score(run_dir: Path) -> float:
    return json.loads((run_dir / "metrics_val.json").read_text())["mean"]["score"]


# ------------------------------------------------------------------- splits

def cut_splits(scene: str) -> dict[str, list[str]]:
    """Cut nth + match-test splits of equal size; return {split: val_ids}."""
    from src.data_prep.make_val_split import (list_images, make_split_match_test,
                                              write_split)
    images_dir = RAW_ROOT / scene / "train" / "images"
    out_root = PROC_ROOT / scene / "splits_exp019"

    names = list_images(images_dir)
    every_n = max(1, round(len(names) / N_VAL))
    val_nth = names[::every_n][:N_VAL]
    write_split(out_root / "nth", [n for n in names if n not in set(val_nth)], val_nth)

    make_split_match_test(images_dir, PROC_ROOT / scene / "colmap_train_only",
                          RAW_ROOT / scene / "test" / "test_poses.csv",
                          out_root / "match", N_VAL)
    val_match = (out_root / "match" / "val_ids.txt").read_text().split()
    return {"nth": val_nth, "match": val_match}


# --------------------------------------------------------- render + scoring

def render_union(run_dir: Path, scene: str, union_ids: set[str]) -> Path:
    from src.utils.render_utils import load_colmap_poses, render_pose_rows
    out = run_dir / "renders_exp019_val"
    rows = load_colmap_poses(PROC_ROOT / scene / "colmap_train_only", only_names=union_ids)
    missing = union_ids - {r["image_name"] for r in rows}
    if missing:  # nth ids need not all be COLMAP-registered
        print(f"    {len(missing)} val ids unregistered in COLMAP, skipped: "
              f"{sorted(missing)[:3]}...")
    render_pose_rows(find_config(run_dir), rows, out)
    return out


def score_renders(render_dir: Path, gt_dir: Path) -> dict[str, dict]:
    """{image_name: metrics row} for every rendered val image."""
    from src.metrics import compute_metrics
    result = compute_metrics(render_dir, gt_dir, lpips_net="alex", psnr_max=50.0)
    (render_dir.parent / "metrics_exp019_val.json").write_text(json.dumps(result, indent=2))
    return {r["image"]: r for r in result["per_image"] if r["image"] != "mean"}


def aggregate_split(per_image: dict[str, dict], ids: list[str]) -> dict:
    """Mean metrics over the split's ids that were actually rendered/scored."""
    rows = [per_image[i] for i in ids if i in per_image]
    if not rows:
        raise RuntimeError("no scored images for split")
    return {"n": len(rows),
            **{k: float(np.mean([r[k] for r in rows]))
               for k in ("psnr", "ssim", "lpips", "score")}}


# ----------------------------------------------------------------- analysis

def spearman(a: list[float], b: list[float]) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def pairwise_agreement(val_scores: list[float], test_scores: list[float],
                       min_delta: float = MIN_TEST_DELTA) -> tuple[int, int]:
    """(n_agree, n_graded) over variant pairs with a non-tie test-GT delta.
    Grades the actual A/B decision: does sign(val delta) match sign(test delta)?"""
    n_agree = n_graded = 0
    for i in range(len(test_scores)):
        for j in range(i + 1, len(test_scores)):
            dt = test_scores[i] - test_scores[j]
            if abs(dt) < min_delta:
                continue
            n_graded += 1
            n_agree += (val_scores[i] - val_scores[j]) * dt > 0
    return n_agree, n_graded


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="list planned work only")
    ap.add_argument("--include-vggt", action="store_true",
                    help="add exp003 (big known loser; inflates rank agreement)")
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    args = ap.parse_args()

    plan = {s: discover_variants(s, args.include_vggt) for s in args.scenes}
    for s, variants in plan.items():
        print(f"{s}: {len(variants)} variants -> {[n for n, _ in variants]}")
    if args.dry_run:
        return

    long_rows, summary_rows = [], []
    for scene in args.scenes:
        splits = cut_splits(scene)
        union = set().union(*splits.values())
        gt_dir = RAW_ROOT / scene / "train" / "images"

        per_variant: dict[str, dict] = {}
        for name, run_dir in plan[scene]:
            print(f"  {scene}/{name}: rendering {len(union)} val poses...")
            render_dir = render_union(run_dir, scene, union)
            per_image = score_renders(render_dir, gt_dir)
            t = test_gt_score(run_dir)
            per_variant[name] = {"test_gt": t}
            for split, ids in splits.items():
                agg = aggregate_split(per_image, ids)
                per_variant[name][split] = agg["score"]
                long_rows.append({"scene": scene, "variant": name, "split": split,
                                  **agg, "test_gt_score": round(t, 5)})

        names = sorted(per_variant)
        test = [per_variant[n]["test_gt"] for n in names]
        for split in ("nth", "match"):
            val = [per_variant[n][split] for n in names]
            agree, graded = pairwise_agreement(val, test)
            summary_rows.append({
                "scene": scene, "split": split, "n_variants": len(names),
                "spearman_vs_test_gt": round(spearman(val, test), 3),
                "ab_agree": agree, "ab_graded": graded,
                "ab_agreement": round(agree / graded, 3) if graded else "",
            })

    for path, rows in ((RESULTS_CSV, long_rows), (SUMMARY_CSV, summary_rows)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}")

    print("\n=== exp019 verdict inputs (higher = better selection signal) ===")
    for r in summary_rows:
        print(f"{r['scene']:>8} {r['split']:>5}: spearman={r['spearman_vs_test_gt']:+.3f} "
              f"A/B agreement={r['ab_agree']}/{r['ab_graded']}")


if __name__ == "__main__":
    main()
