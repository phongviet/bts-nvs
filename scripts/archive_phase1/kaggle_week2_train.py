#!/usr/bin/env python
"""Rented-GPU (Kaggle) runner for the locked Week-2 exp005 config.

Per scene: build a train_staging_dense-style symlink staging dir from the
uploaded dataset -> ns-train splatfacto (dense-COLMAP init + antialiased,
30k iters) -> render every pose in test/test_poses.csv with src/render.py.
Mirrors scripts/run_final_private_antialiased.sh: same train args, same
completion check (last step-*.ckpt >= max_iters-1), same incomplete-run
removal, same renders_test/.done marker, so each output scene dir is a
drop-in runs/phase1/exp005_antialiased_dense/<scene>/ run dir.

Expected input layout (the kaggle-upload-week2-train.zip dataset):
    <input-root>/<split>/<scene>/train/images/*.JPG
    <input-root>/<split>/<scene>/colmap_dense_init/{cameras,images,points3D}.bin
    <input-root>/<split>/<scene>/test/test_poses.csv

Invocation (from the notebook, inside the airace env with ns-train on PATH):
    python scripts/kaggle_week2_train.py \
        --input-root /kaggle/input/<dataset>/kaggle_upload/phase1 \
        --output-root /kaggle/working/week2_train_output \
        --scenes private_set1/HCM1439 private_set1/HNI0131 ...

Quiet mode: ns-train/render output goes to per-scene log files under
--log-root; only one-line statuses print, with the log tail surfaced on
failure (same convention as the dense-COLMAP Kaggle notebook).
"""
import argparse
import glob
import os
import queue
import shutil
import subprocess
import sys
import threading
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RENDER_PY = REPO_ROOT / "src" / "render.py"


def visible_gpu_count():
    """Number of CUDA devices, honoring an already-set CUDA_VISIBLE_DEVICES."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return len([d for d in cvd.split(",") if d.strip() != ""])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        return len([l for l in out.stdout.splitlines() if l.startswith("GPU ")])
    except FileNotFoundError:
        return 0


def run_cmd(cmd, log_path, dry_run=False, env=None):
    print(f"+ {' '.join(cmd)}  (log: {log_path})", flush=True)
    if dry_run:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0:
        with open(log_path, errors="replace") as logf:
            tail = logf.readlines()[-40:]
        print(f"!! command failed (exit {result.returncode}), last 40 lines of {log_path}:", flush=True)
        print("".join(tail), flush=True)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    print("  done.", flush=True)


def last_ckpt_step(run_dir):
    ckpts = sorted(glob.glob(os.path.join(run_dir, "**", "step-*.ckpt"), recursive=True))
    return int(Path(ckpts[-1]).stem.split("-")[-1]) if ckpts else None


def make_staging(scene_dir, staging_root, scene):
    # Same layout build_dense_colmap.make_train_dir() produces locally:
    # staging/{images, sparse/0} symlinks (reads from a read-only input
    # dataset are fine). Named train_staging_dense so the ns-train run tree
    # matches the local exp005 runs.
    staging = os.path.join(staging_root, scene, "train_staging_dense")
    os.makedirs(os.path.join(staging, "sparse"), exist_ok=True)
    for link, target in [
        (os.path.join(staging, "images"), os.path.join(scene_dir, "train", "images")),
        (os.path.join(staging, "sparse", "0"), os.path.join(scene_dir, "colmap_dense_init")),
    ]:
        if os.path.islink(link):
            os.unlink(link)
        os.symlink(target, link)
    return staging


def process_scene(rel_scene_path, args, gpu_id=None):
    scene = os.path.basename(rel_scene_path)
    scene_dir = os.path.join(args.input_root, rel_scene_path)
    run_dir = os.path.join(args.output_root, scene)
    renders = os.path.join(run_dir, "renders_test")
    log_dir = os.path.join(args.log_root, scene)
    env = None
    if gpu_id is not None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    if os.path.exists(os.path.join(renders, ".done")):
        print(f"[{scene}] renders_test/.done present, skipping.")
        return
    for sub in ("train/images", "colmap_dense_init", "test/test_poses.csv"):
        if not os.path.exists(os.path.join(scene_dir, sub)):
            raise FileNotFoundError(f"{scene_dir}/{sub} missing from the input dataset")

    staging = make_staging(scene_dir, args.staging_root, scene)

    step = last_ckpt_step(run_dir)
    if step is not None and step >= args.max_iters - 1:
        print(f"[{scene}] training already complete (step {step}), skipping training.")
    else:
        if os.path.isdir(run_dir) and not args.dry_run:
            print(f"[{scene}] incomplete run (last step: {step}), removing and retraining from scratch.")
            shutil.rmtree(run_dir)
        run_cmd([
            "ns-train", "splatfacto",
            "--data", staging,
            "--output-dir", run_dir,
            "--max-num-iterations", str(args.max_iters),
            "--viewer.quit-on-train-completion", "True",
            "--pipeline.model.rasterize-mode", "antialiased",
            "colmap", "--eval-mode", "all", "--colmap-path", "sparse/0",
        ], os.path.join(log_dir, "ns_train.log"), args.dry_run, env=env)

    configs = sorted(glob.glob(os.path.join(run_dir, "**", "config.yml"), recursive=True))
    config = configs[-1] if configs else os.path.join(run_dir, "<config.yml after training>")
    run_cmd([
        sys.executable, str(RENDER_PY),
        "--config", config, "--mode", "test",
        "--poses-csv", os.path.join(scene_dir, "test", "test_poses.csv"),
        "--out", renders,
    ], os.path.join(log_dir, "render.log"), args.dry_run, env=env)
    if args.dry_run:
        return
    Path(renders, ".done").touch()

    if args.strip_checkpoints:
        for ckpt in glob.glob(os.path.join(run_dir, "**", "step-*.ckpt"), recursive=True):
            os.remove(ckpt)
    n = len([p for p in os.listdir(renders) if p != ".done"])
    print(f"[{scene}] DONE -- {n} test renders in {renders}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", required=True, help="dataset phase1/ dir, e.g. /kaggle/input/<ds>/kaggle_upload/phase1")
    ap.add_argument("--output-root", required=True, help="e.g. /kaggle/working/week2_train_output")
    ap.add_argument("--scenes", nargs="+", required=True, help="split-relative, e.g. private_set1/HCM1439")
    ap.add_argument("--max-iters", type=int, default=30000)
    ap.add_argument("--staging-root", default="/kaggle/working/_staging")
    ap.add_argument("--log-root", default="/kaggle/working/_logs")
    ap.add_argument("--strip-checkpoints", action="store_true",
                    help="drop step-*.ckpt from outputs (smaller zip, but the local "
                         "fleet script will RETRAIN such scenes; exp009 also needs the ckpts)")
    ap.add_argument("--dry-run", action="store_true", help="print the per-scene commands without running")
    ap.add_argument("--parallel", default="1",
                    help="concurrent scenes, one GPU each ('auto' = one per visible GPU; "
                         "Kaggle T4 x2 sessions -> 2)")
    args = ap.parse_args()

    workers = visible_gpu_count() if args.parallel == "auto" else int(args.parallel)
    workers = max(1, workers)

    print(f"{len(args.scenes)} scenes to process: {args.scenes}", flush=True)
    failed = []

    if workers == 1:
        for rel in args.scenes:
            print(f"=============== {rel} ===============", flush=True)
            try:
                process_scene(rel, args)
            except Exception as e:
                print(f"!! FAILED {rel}: {e}", flush=True)
                traceback.print_exc()
                failed.append(rel)
    else:
        # One worker thread per GPU, pinned via CUDA_VISIBLE_DEVICES, pulling
        # scenes off a shared queue (threads are fine -- all heavy work runs
        # in subprocesses, and per-scene output already goes to log files).
        print(f"=== running up to {workers} scenes in parallel, one per GPU ===", flush=True)
        scene_q = queue.Queue()
        for rel in args.scenes:
            scene_q.put(rel)

        def worker(gpu_id):
            while True:
                try:
                    rel = scene_q.get_nowait()
                except queue.Empty:
                    return
                print(f"=============== {rel} [gpu{gpu_id}] ===============", flush=True)
                try:
                    process_scene(rel, args, gpu_id=gpu_id)
                except Exception as e:
                    print(f"!! FAILED {rel} [gpu{gpu_id}]: {e}", flush=True)
                    traceback.print_exc()
                    failed.append(rel)

        threads = [threading.Thread(target=worker, args=(g,)) for g in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    if failed:
        raise SystemExit(f"failed scenes: {failed}")
    print("All scenes complete.", flush=True)


if __name__ == "__main__":
    main()
