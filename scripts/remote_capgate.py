"""Remote capacity gate for one drone scene (HCM0421), backbone-only.

The question: splatfacto-big culls hard. bonsai lands at 0.25 gauss/px (starved -> SSS won
+0.296 LB); chair at 0.89 (saturated -> SSS lost -0.133). No drone backbone has ever been
measured. If the drones cull into starved territory, capacity is the one lever with a large
precedent on this project, and the 5 drones carry 71% of the leaderboard.

Three arms, scored on the SAME 25-frame match-test hold-out (backbone-only renders, no DIBR
/refiner -- this is a first filter: if capacity does not move the raw render it will not move
the stack, and fidelity gains carry ~70-94% through the refiner while LPIPS carries ~26%):

  ctrl   splatfacto-big, 30k, antialiased            == the shipped drone backbone
  relax  + lower cull + denser splits, longer split  == more retained capacity, single axis
  long   splatfacto-big, 100k, antialiased           == the exp006 "busy scenes gain at 100k"

Each arm prints gauss/px, which is the diagnostic that decides everything: near bonsai's 0.25
=> starved => worth a full 5-scene retrain; near chair's 0.89 => saturated => dead.

Run (inside the torch 2.5.1 / cu121 image, after provision.sh):
    python remote_capgate.py --arm ctrl   --data /workspace/data/HCM0421 --out /workspace/out
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # payload root: src/ lives here

ITERS = {"ctrl": 30000, "relax": 30000, "long": 100000}


def train(arm, data, out):
    run_out = out / arm
    cfgs = list(run_out.glob("train/splatfacto*/*/config.yml"))
    if cfgs:
        print(f"[{arm}] checkpoint exists, skipping train", flush=True)
        return cfgs[0]
    cmd = ["ns-train", "splatfacto-big",
           "--data", str(data / "train"),
           "--output-dir", str(run_out),
           "--experiment-name", "train",
           "--max-num-iterations", str(ITERS[arm]),
           "--pipeline.model.rasterize-mode", "antialiased",
           "--viewer.quit-on-train-completion", "True"]
    if arm == "relax":
        # single axis: keep more primitives. Lower the opacity cull, split on a smaller
        # gradient, and keep densifying later into training. Everything else == ctrl.
        cmd += ["--pipeline.model.cull-alpha-thresh", "0.001",
                "--pipeline.model.densify-grad-thresh", "0.0002",
                "--pipeline.model.stop-split-at", "27000"]
    if arm == "long":
        # a FAIR 100k iters test: splatfacto-big hardcodes the means LR schedule to
        # max_steps=30000, so without this the last 70k iters run at the frozen minimum
        # LR (refinement, not the exp006 "busy scenes still gain at 100k" signal). Extend
        # the means schedule to match, and densify later so capacity actually grows.
        cmd += ["--optimizers.means.scheduler.max-steps", "100000",
                "--pipeline.model.stop-split-at", "50000"]
    cmd += ["colmap", "--eval-mode", "all", "--colmap-path", "sparse/0",
            "--downscale-factor", "1"]
    print(f"[{arm}] {' '.join(cmd)}", flush=True)
    log = run_out / "train.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        p.wait()
    assert p.returncode == 0, f"[{arm}] ns-train exited {p.returncode} -- see {log}"
    cfgs = list(run_out.glob("train/splatfacto*/*/config.yml"))
    assert len(cfgs) == 1, f"[{arm}] expected 1 config, found {len(cfgs)}"
    return cfgs[0]


def n_gaussians(cfg):
    import torch
    ck = sorted(cfg.parent.glob("nerfstudio_models/step-*.ckpt"))[-1]
    sd = torch.load(ck, map_location="cpu", weights_only=False)["pipeline"]
    return int(sd[next(k for k in sd if k.endswith("means"))].shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ITERS))
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from src.utils.render_utils import load_colmap_poses, render_pose_rows
    from src.metrics import compute_metrics
    from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary

    cfg = train(args.arm, args.data, args.out)

    val_ids = set((args.data / "val_ids.txt").read_text().split())
    rows = load_colmap_poses(args.data / "val/sparse/0", only_names=val_ids)
    rend = args.out / args.arm / "renders_val"
    n = render_pose_rows(cfg, rows, rend)
    assert n == 25 or len(list(rend.iterdir())) == 25, f"[{args.arm}] rendered {n} of 25"

    res = compute_metrics(rend, args.data / "val/images", "vgg", 50.0)
    m = res["mean"]
    cam = list(read_cameras_binary(args.data / "train/sparse/0/cameras.bin").values())[0]
    ng = n_gaussians(cfg)
    m["n_gaussians"] = ng
    m["gauss_per_px"] = ng / (cam.width * cam.height)
    m["iters"] = ITERS[args.arm]
    (args.out / args.arm / "metrics.json").write_text(json.dumps(res, indent=2))
    print(f"\n[{args.arm}] iters={ITERS[args.arm]}  gaussians={ng:,} "
          f"({m['gauss_per_px']:.3f} gauss/px)", flush=True)
    print(f"[{args.arm}] PSNR {m['psnr']:.3f}  SSIM {m['ssim']:.4f}  "
          f"LPIPS {m['lpips']:.4f}  Score {m['score']:.5f}", flush=True)


if __name__ == "__main__":
    main()
