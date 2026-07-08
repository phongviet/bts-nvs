"""Config-driven experiment sweep runner (Tier-A ablations, exp006-014).

Reads a sweep YAML from configs/experiments/, runs scene x variant training
with ns-train, renders + scores, and appends one CSV row per completed cell:
    scene,exp,variant,psnr,ssim,lpips,score,train_hours,gpu
Cells already present in the CSV are skipped (rerun with --force to redo).

Sweep YAML schema:
    experiment: exp006_capacity_iters_sweep
    results_csv: results/week3_tierA_ablation.csv
    gpu: rented | local          # provenance only, recorded in the CSV
    split: public_set            # raw-data split the scenes live in
    scenes: [hcm0034, HCM0181]
    data_template: "data/processed/phase1/{scene}/train_staging_dense"
    eval: public_gt | val_split  # public scenes have test GT; private use the
                                 # match-test val split (see src/render_val.py)
    base_args: ["--pipeline.model.rasterize-mode antialiased"]
    variants:
      big_30k: {method: splatfacto-big, iters: 30000}
      mcmc2M_30k:
        method: splatfacto-mcmc
        iters: 30000
        args: ["--pipeline.model.cap-max", "2000000"]
        load_dir_template: "runs/.../{scene}/..."   # optional, for fine-tunes

Custom methods (splatfacto-mcmc / -perceptual / -tpw) are registered via
NERFSTUDIO_METHOD_CONFIGS automatically -- no manual export needed.

Usage:
    python scripts/run_sweep.py --sweep configs/experiments/exp006_capacity_iters_sweep.yaml
    python scripts/run_sweep.py --sweep ... --scenes hcm0034 --variants big_30k --dry-run
"""
import argparse
import csv
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CSV_FIELDS = ["scene", "exp", "variant", "psnr", "ssim", "lpips", "score",
              "train_hours", "gpu", "timestamp"]
CUSTOM_METHODS = ["splatfacto-mcmc", "splatfacto-perceptual",
                  "splatfacto-mcmc-perceptual", "splatfacto-tpw"]
METHOD_CONFIGS_ENV = ",".join(
    f"{m}=src.register_custom_methods:{m.replace('-', '_')}_method" for m in CUSTOM_METHODS)


def load_sweep(path: Path) -> dict:
    sweep = yaml.safe_load(path.read_text())
    for key in ("experiment", "results_csv", "scenes", "variants", "data_template",
                "eval", "split", "gpu"):
        if key not in sweep:
            raise SystemExit(f"{path}: missing required key '{key}'")
    if sweep["eval"] not in ("public_gt", "val_split"):
        raise SystemExit(f"{path}: eval must be public_gt or val_split")
    return sweep


def run_dir_for(sweep: dict, scene: str, variant: str) -> Path:
    return REPO_ROOT / "runs" / "phase1" / sweep["experiment"] / scene / variant


def build_train_cmd(sweep: dict, variant_name: str, scene: str) -> list[str]:
    v = sweep["variants"][variant_name]
    data = sweep["data_template"].format(scene=scene)
    cmd = ["ns-train", v["method"],
           "--data", data,
           "--output-dir", str(run_dir_for(sweep, scene, variant_name)),
           "--max-num-iterations", str(v["iters"]),
           "--viewer.quit-on-train-completion", "True"]
    if "load_dir_template" in v:
        cmd += ["--load-dir", v["load_dir_template"].format(scene=scene)]
    cmd += list(sweep.get("base_args", [])) + [str(a).format(scene=scene)
                                               for a in v.get("args", [])]
    cmd += ["colmap", "--eval-mode", "all", "--colmap-path", "sparse/0"]
    return cmd


def training_complete(run_dir: Path, iters: int) -> bool:
    ckpts = sorted(run_dir.glob("**/step-*.ckpt"))
    if not ckpts:
        return False
    last = int(ckpts[-1].stem.split("-")[-1])
    return last >= iters - 1


def find_config(run_dir: Path) -> Path:
    configs = sorted(run_dir.glob("**/config.yml"))
    if not configs:
        raise SystemExit(f"no config.yml under {run_dir} -- training failed?")
    return configs[-1]


def existing_cells(csv_path: Path) -> set[tuple[str, str, str]]:
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {(r["scene"], r["exp"], r["variant"]) for r in csv.DictReader(f)}


def append_row(csv_path: Path, row: dict):
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def evaluate(sweep: dict, scene: str, variant: str, env: dict) -> dict:
    """Render + score one trained cell; returns the mean-metrics dict."""
    import json
    run_dir = run_dir_for(sweep, scene, variant)
    config = find_config(run_dir)
    scene_dir = REPO_ROOT / "data" / "raw" / "phase1" / sweep["split"] / scene
    metrics_json = run_dir / "metrics_val.json"

    if sweep["eval"] == "public_gt":
        renders = run_dir / "renders_test"
        subprocess.run([sys.executable, "src/render.py", "--config", str(config),
                        "--mode", "test", "--poses-csv",
                        str(scene_dir / "test" / "test_poses.csv"),
                        "--out", str(renders)], cwd=REPO_ROOT, env=env, check=True)
        subprocess.run([sys.executable, "src/metrics.py", "--renders", str(renders),
                        "--gt", str(scene_dir / "test" / "images"),
                        "--out", str(metrics_json)], cwd=REPO_ROOT, env=env, check=True)
    else:
        subprocess.run([sys.executable, "src/render_val.py", "--config", str(config),
                        "--scene-dir", str(scene_dir),
                        "--processed-root", "data/processed/phase1",
                        "--out", str(run_dir / "renders_val_split"),
                        "--metrics-out", str(metrics_json)],
                       cwd=REPO_ROOT, env=env, check=True)

    d = json.loads(metrics_json.read_text())
    per = d["per_image"]
    return {k: sum(x[k] for x in per) / len(per) for k in ("psnr", "ssim", "lpips", "score")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, type=Path)
    ap.add_argument("--scenes", nargs="*", help="subset of the sweep's scenes")
    ap.add_argument("--variants", nargs="*", help="subset of the sweep's variants")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    ap.add_argument("--force", action="store_true", help="redo cells already in the CSV")
    args = ap.parse_args()

    sweep = load_sweep(args.sweep)
    scenes = args.scenes or sweep["scenes"]
    variants = args.variants or list(sweep["variants"])
    for s in scenes:
        if s not in sweep["scenes"]:
            raise SystemExit(f"scene {s} not in sweep {sweep['experiment']}")
    for v in variants:
        if v not in sweep["variants"]:
            raise SystemExit(f"variant {v} not in sweep {sweep['experiment']}")

    csv_path = REPO_ROOT / sweep["results_csv"]
    done = set() if args.force else existing_cells(csv_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env["NERFSTUDIO_METHOD_CONFIGS"] = METHOD_CONFIGS_ENV

    for scene in scenes:
        for variant in variants:
            key = (scene, sweep["experiment"], variant)
            if key in done:
                print(f"== {scene}/{variant}: already in {csv_path.name}, skipping ==")
                continue
            cmd = build_train_cmd(sweep, variant, scene)
            if args.dry_run:
                print(f"[dry-run] {scene}/{variant}:\n  {' '.join(cmd)}")
                continue

            data_dir = REPO_ROOT / sweep["data_template"].format(scene=scene)
            if not data_dir.exists():
                print(f"!! {scene}/{variant}: {data_dir} missing -- build init/staging first. Skipping.")
                continue

            run_dir = run_dir_for(sweep, scene, variant)
            iters = sweep["variants"][variant]["iters"]
            train_hours = 0.0
            if training_complete(run_dir, iters):
                print(f"== {scene}/{variant}: training complete, skipping to eval ==")
            else:
                print(f"== {scene}/{variant}: training ({iters} iters) ==")
                t0 = time.time()
                subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
                train_hours = (time.time() - t0) / 3600

            m = evaluate(sweep, scene, variant, env)
            row = {"scene": scene, "exp": sweep["experiment"], "variant": variant,
                   **{k: f"{m[k]:.4f}" for k in ("psnr", "ssim", "lpips", "score")},
                   "train_hours": f"{train_hours:.2f}", "gpu": sweep["gpu"],
                   "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
            append_row(csv_path, row)
            print(f"== {scene}/{variant}: score={m['score']:.4f} -> {csv_path} ==")


if __name__ == "__main__":
    main()
