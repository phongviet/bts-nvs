"""SSS public-scene fleet driver for Kaggle (T4x2 or P100).

Per scene: undistort (SIMPLE_RADIAL -> PINHOLE, test frames filtered) ->
SSS train (paper-scale params) -> render official test poses -> score with
the competition metric (LPIPS-vgg, PSNR_max=50). Scenes run in parallel,
one per visible GPU. Idempotent: a scene with metrics_test.json is skipped,
so a 12h-session timeout never loses completed scenes.

Usage (from the SSS repo checkout dir; see the notebook):
  python kaggle_sss_fleet.py --dataset /kaggle/input/<slug>/kaggle_upload \
      --out /kaggle/working/sss_out --scenes hcm0031 hcm0034 ... \
      --cap-max 2000000 --iters 40000 [--keep-ply]
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SSS_REPO = HERE / "3D-student-splatting-and-scooping"


def run(cmd, env=None, log_path=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    e = dict(os.environ)
    if env:
        e.update(env)
    with open(log_path, "ab") if log_path else open(os.devnull, "wb") as lf:
        p = subprocess.run(list(map(str, cmd)), env=e, stdout=lf, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        if log_path:
            print(Path(log_path).read_text()[-3000:], flush=True)
        raise RuntimeError(f"FAILED ({p.returncode}): {' '.join(map(str, cmd))}")


def do_scene(scene, args, gpu):
    t0 = time.time()
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    sdir = Path(args.dataset) / "data" / scene
    out = Path(args.out) / scene
    out.mkdir(parents=True, exist_ok=True)
    log = out / "train.log"
    metrics_json = out / "metrics_test.json"
    if metrics_json.exists():
        print(f"[{scene}] already scored -- skip", flush=True)
        return scene, json.load(open(metrics_json))["mean"]

    # 1. undistort to pinhole (also filters test frames out of images.bin)
    undist = Path(args.out) / "_undist" / scene
    if not (undist / "sparse/0/cameras.bin").exists():
        run([sys.executable, HERE / "make_undistorted_scene.py",
             "--src", sdir, "--dst", undist], env=env, log_path=log)

    # 2. train (repo default eval=False after our patch: all train images used)
    # Rolling checkpoints every 2500 iters + auto-resume: the post-burnin SGHMC
    # noise phase can (rarely, stochastically) still crash the rasterizer; a
    # resume re-rolls the noise from the last checkpoint instead of losing the
    # scene (observed on the first T4 fleet run: 4/5 scenes died at ~7.3-7.8k).
    model = out / "model"
    ply = model / f"point_cloud/iteration_{args.iters}/point_cloud.ply"
    ckpt_iters = [str(i) for i in range(2500, args.iters, 2500)]
    for attempt in range(4):
        if ply.exists():
            break
        cmd = [sys.executable, SSS_REPO / "train.py", "-s", undist, "-m", model,
               "--data_device", args.data_device,
               "--cap_max", args.cap_max, "--nu_degree", 100,
               "--C_burnin", 5e5, "--C", 1.2e2, "--burnin_iterations", 7000,
               "--iterations", args.iters, "--save_iterations", args.iters,
               "--test_iterations", args.iters, "--quiet",
               "--checkpoint_iterations", *ckpt_iters]
        ckpts = sorted(model.glob("chkpnt*.pth"),
                       key=lambda p: int(p.stem.replace("chkpnt", "")))
        if ckpts:
            print(f"[{scene}] resuming from {ckpts[-1].name} "
                  f"(attempt {attempt + 1})", flush=True)
            cmd += ["--start_checkpoint", ckpts[-1]]
        try:
            run(cmd, env=env, log_path=log)
        except RuntimeError:
            if attempt == 3:
                raise
    for c in model.glob("chkpnt*.pth"):
        c.unlink()  # multi-GB; no longer needed once the final ply exists

    # 3. render the official test poses
    renders = out / "renders_test"
    run([sys.executable, HERE / "render_test_csv.py", "--model", model,
         "--poses-csv", sdir / "test_poses.csv", "--out", renders],
        env=env, log_path=log)

    # 4. score vs test GT with the competition metric
    run([sys.executable, HERE / "metrics.py", "--renders", renders,
         "--gt", sdir / "test_gt", "--out", metrics_json],
        env=env, log_path=log)

    if not args.keep_ply:
        ply.unlink()  # ~0.5GB per scene at 2M comps; renders+metrics suffice
    mean = json.load(open(metrics_json))["mean"]
    print(f"[{scene}] DONE in {(time.time()-t0)/60:.0f} min: {mean}", flush=True)
    return scene, mean


def worker(q, results, args, gpu):
    while True:
        try:
            scene = q.get_nowait()
        except Exception:
            return
        try:
            results.append(do_scene(scene, args, gpu))
        except Exception as ex:  # keep the fleet going; scene is retryable
            print(f"[{scene}] ERROR: {ex}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--cap-max", type=int, default=2_000_000)
    ap.add_argument("--iters", type=int, default=40_000)
    ap.add_argument("--data-device", default="cpu")
    ap.add_argument("--keep-ply", action="store_true")
    args = ap.parse_args()

    import torch
    n_gpu = torch.cuda.device_count()
    print(f"GPUs: {n_gpu}, scenes: {args.scenes}, cap_max={args.cap_max}, "
          f"iters={args.iters}", flush=True)

    mgr = mp.Manager()
    q = mgr.Queue()
    results = mgr.list()
    for s in args.scenes:
        q.put(s)
    procs = [mp.Process(target=worker, args=(q, results, args, g))
             for g in range(max(1, n_gpu))]
    [p.start() for p in procs]
    [p.join() for p in procs]

    summary = {s: m for s, m in results}
    done = [s for s in args.scenes if s in summary]
    out = Path(args.out)
    if done:
        summary["MEAN"] = {k: sum(summary[s][k] for s in done) / len(done)
                           for k in ("psnr", "ssim", "lpips", "psnr_norm", "score")}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    missing = [s for s in args.scenes if s not in summary]
    if missing:
        print("INCOMPLETE, rerun for:", missing, flush=True)


if __name__ == "__main__":
    main()
