"""exp034 Kaggle fleet driver — the FINAL PLAN for private scenes, per scene:

  1. stage: rebuild the repo-relative paths that Warper (04) + ns-train expect
     from the flat uploaded dataset, via symlinks:
       data/raw/phase1/private_set1/<scene>/train/images      -> <ds>/train_images
       data/raw/phase1/private_set1/<scene>/train/sparse/0     -> <ds>/raw_sparse0   (SIMPLE_RADIAL, k)
       data/raw/phase1/private_set1/<scene>/test/test_poses.csv-> <ds>/test_poses.csv
       data/processed/phase1/<scene>/train_staging_dense/images   -> <ds>/train_images
       data/processed/phase1/<scene>/train_staging_dense/sparse/0 -> <ds>/dense_sparse0 (dense init)
  2. train the big backbone (E4/E6): ns-train splatfacto-big, dense init, 30k, antialiased.
  3. refiner v2 stack (E3+E5): 10_refiner_pilot.py --config <big run> --ss 2 --sample cubic
     --base 48 --iters 6000 --ema 0.999 --tta --png --max-pairs N  -> PNG test renders + refiner_v2.pt.
  4. collect: copy renders + ckpt + val_loss into OUT_ROOT/<scene>/.

Idempotent + resumable: skips a scene whose refined PNGs already exist; skips big
training when a 30k ckpt is present; refiner caches pairs/inputs. Package after
EVERY scene so a session timeout never loses finished scenes.

Run (inside a Kaggle notebook, airace env on PATH):
  python Analysis/kaggle_exp034_fleet.py --dataset /kaggle/input/<slug>/kaggle_upload \
      --scenes HCM0249 HCM0254 --out /kaggle/working/exp034_out [--parallel auto] [--max-iters 30000]
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW_PRIV = REPO / "data/raw/phase1/private_set1"
PROC = REPO / "data/processed/phase1"
REFINER_OUT = REPO / "Analysis/X5_refiner"
PY = sys.executable


def link(target: Path, linkname: Path):
    linkname.parent.mkdir(parents=True, exist_ok=True)
    if linkname.is_symlink() or linkname.exists():
        if linkname.is_symlink() or linkname.is_file():
            linkname.unlink()
        else:
            shutil.rmtree(linkname)
    os.symlink(target, linkname)


def stage(scene: str, ds: Path):
    sd = ds / scene
    for p in ("train_images", "raw_sparse0/cameras.bin", "raw_sparse0/images.bin",
              "dense_sparse0/points3D.bin", "test_poses.csv"):
        assert (sd / p).exists(), f"dataset missing {scene}/{p}"
    raw = RAW_PRIV / scene
    link(sd / "train_images", raw / "train/images")
    link(sd / "raw_sparse0", raw / "train/sparse/0")
    link(sd / "test_poses.csv", raw / "test/test_poses.csv")
    dense = PROC / scene / "train_staging_dense"
    link(sd / "train_images", dense / "images")
    link(sd / "dense_sparse0", dense / "sparse/0")
    return raw, dense


def big_ckpt_step(run_dir: Path):
    ckpts = sorted(run_dir.glob("**/nerfstudio_models/step-*.ckpt"))
    if not ckpts:
        return None
    return int(ckpts[-1].stem.split("-")[-1])


def train_big(scene: str, dense: Path, run_dir: Path, max_iters: int, gpu: int | None, logf: Path):
    step = big_ckpt_step(run_dir)
    if step is not None and step >= max_iters - 1:
        print(f"[{scene}] big backbone present (step {step}); skip training.", flush=True)
        return
    if run_dir.exists():
        shutil.rmtree(run_dir)
    env = {**os.environ}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = ["ns-train", "splatfacto-big", "--data", str(dense),
           "--output-dir", str(run_dir), "--max-num-iterations", str(max_iters),
           "--viewer.quit-on-train-completion", "True",
           "--pipeline.model.rasterize-mode", "antialiased",
           "colmap", "--eval-mode", "all", "--colmap-path", "sparse/0"]
    print(f"[{scene}] + {' '.join(cmd)}", flush=True)
    logf.parent.mkdir(parents=True, exist_ok=True)
    with open(logf, "w") as f:
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    assert r.returncode == 0, f"[{scene}] ns-train failed -- see {logf}"


def refiner_v2(scene: str, run_dir: Path, max_pairs: int, iters: int, gpu: int | None, logf: Path):
    env = {**os.environ}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [PY, str(REPO / "Analysis/10_refiner_pilot.py"), "--scene", scene,
           "--config", str(run_dir), "--ss", "2", "--sample", "cubic",
           "--base", "48", "--iters", str(iters), "--ema", "0.999",
           "--tta", "--png", "--suffix", "_v2", "--max-pairs", str(max_pairs)]
    print(f"[{scene}] + {' '.join(cmd)}", flush=True)
    logf.parent.mkdir(parents=True, exist_ok=True)
    with open(logf, "w") as f:
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    assert r.returncode == 0, f"[{scene}] refiner failed -- see {logf}"


def collect(scene: str, out_root: Path, logf: Path):
    dst = out_root / scene
    dst.mkdir(parents=True, exist_ok=True)
    renders = REFINER_OUT / scene / "renders_refined_v2"
    pngs = sorted(renders.glob("*.png"))
    assert pngs, f"[{scene}] no refined PNGs at {renders}"
    rdst = dst / "renders_refined_v2"
    if rdst.exists():
        shutil.rmtree(rdst)
    shutil.copytree(renders, rdst)
    ckpt = REFINER_OUT / scene / "refiner_v2.pt"
    if ckpt.exists():
        shutil.copy2(ckpt, dst / "refiner_v2.pt")
    vloss = ""
    if logf.exists():
        for ln in logf.read_text(errors="ignore").splitlines():
            if "best val_loss" in ln:
                vloss = ln.strip()
    (dst / "STATUS.txt").write_text(f"{scene}: {len(pngs)} refined PNGs\n{vloss}\n")
    print(f"[{scene}] DONE -- {len(pngs)} PNGs -> {rdst}  {vloss}", flush=True)


def purge_scratch(scene: str, big_root: Path):
    """Free per-scene intermediates once results are safely in out_root.
    The big nerfstudio run (~1-2 GB) and the DIBR pair caches / test_inputs
    (~4-6 GB) are pure scratch -- on Kaggle's ~20 GB working disk, leaving them
    around fills the disk after 2-3 scenes (Errno 28). Results (PNGs + ckpt) are
    already copied to out_root by collect(), so this is safe and keeps peak
    footprint at ~2 in-flight scenes regardless of how many total."""
    for p in (big_root / scene, REFINER_OUT / scene):
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            print(f"[{scene}] purged scratch {p}", flush=True)


def process(scene: str, ds: Path, out_root: Path, max_iters: int, max_pairs: int,
            refiner_iters: int, gpu: int | None, big_root: Path, log_root: Path,
            keep_intermediates: bool = False):
    if (out_root / scene / "renders_refined_v2").exists() and \
       list((out_root / scene / "renders_refined_v2").glob("*.png")):
        print(f"[{scene}] already collected; skip.", flush=True)
        return
    raw, dense = stage(scene, ds)
    run_dir = big_root / scene
    train_big(scene, dense, run_dir, max_iters, gpu, log_root / scene / "ns_train.log")
    rlog = log_root / scene / "refiner.log"
    refiner_v2(scene, run_dir, max_pairs, refiner_iters, gpu, rlog)
    collect(scene, out_root, rlog)
    if not keep_intermediates:
        purge_scratch(scene, big_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="kaggle_upload dir (has data/<scene>/...)")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", default="/kaggle/working/exp034_out")
    ap.add_argument("--max-iters", type=int, default=30000)
    ap.add_argument("--max-pairs", type=int, default=90)
    ap.add_argument("--refiner-iters", type=int, default=6000)
    ap.add_argument("--parallel", default="1", help="'auto' = one scene per visible GPU")
    ap.add_argument("--keep-intermediates", action="store_true",
                    help="do NOT purge big run + pair caches after each scene "
                         "(default purges to survive Kaggle's ~20 GB disk)")
    args = ap.parse_args()

    ds = Path(args.dataset) / "data"
    assert ds.is_dir(), f"{ds} not found (expected <dataset>/data/<scene>/...)"
    out_root = Path(args.out)
    big_root = Path("/kaggle/working/exp034_big")
    log_root = Path("/kaggle/working/exp034_logs")

    if args.parallel == "auto":
        try:
            import torch
            workers = max(1, torch.cuda.device_count())
        except Exception:
            workers = 1
    else:
        workers = int(args.parallel)

    print(f"exp034 fleet: {len(args.scenes)} scenes, {workers} worker(s)", flush=True)
    if workers <= 1:
        for s in args.scenes:
            process(s, ds, out_root, args.max_iters, args.max_pairs,
                    args.refiner_iters, None, big_root, log_root,
                    args.keep_intermediates)
    else:
        def job(i_s):
            i, s = i_s
            process(s, ds, out_root, args.max_iters, args.max_pairs,
                    args.refiner_iters, i % workers, big_root, log_root,
                    args.keep_intermediates)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(job, enumerate(args.scenes)))

    print("\nfleet complete. collected scenes:", flush=True)
    for s in args.scenes:
        st = out_root / s / "STATUS.txt"
        print("  " + (st.read_text().strip().replace("\n", " | ") if st.exists() else f"{s}: MISSING"))


if __name__ == "__main__":
    main()
