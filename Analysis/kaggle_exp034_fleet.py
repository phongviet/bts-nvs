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
REFINER_OUT = REPO / "Analysis/X5_refiner"
PY = sys.executable

# Where stage() rebuilds the repo-relative tree. Round 2 must land under
# VAI_NVS_DATA_ROUND2 (not a phase1 split) because 04_x3_dibr_pilot.scene_raw()
# resolves round-2 scenes by looking there -- staging them anywhere else makes
# the DIBR step silently fail to find the scene.
PHASES = {
    "phase1": (REPO / "data/raw/phase1/private_set1", REPO / "data/processed/phase1"),
    "round2": (REPO / "data/raw/VAI_NVS_DATA_ROUND2", REPO / "data/processed/round2"),
}
RAW_PRIV, PROC = PHASES["phase1"]  # rebound by main() per --phase

# Scenes that MUST use plain `splatfacto` instead of `splatfacto-big`.
# bonsai is 1920x1080 -- the highest-pixel-count round-2 scene (2.25x chair,
# ~1.6x a drone frame) -- and splatfacto-big deterministically OOMs it on the
# T4 (16 GB); it failed ns-train on two separate sessions. Gate A validated
# plain splatfacto for bonsai, so we fall back to it by default. Additional
# scenes can be added at the CLI via --base-scenes (merged with this set).
DEFAULT_BASE_SCENES = {"bonsai"}


def link(target: Path, linkname: Path):
    linkname.parent.mkdir(parents=True, exist_ok=True)
    if linkname.is_symlink() or linkname.exists():
        # GUARD: on Kaggle these paths do not exist and we are just rebuilding
        # the tree. Run the same driver on the LOCAL box (e.g. --phase round2,
        # whose roots are the real dataset) and this branch would rmtree the
        # actual competition data. Only ever replace our own symlinks.
        if not linkname.is_symlink():
            raise SystemExit(
                f"refusing to replace real path {linkname} with a symlink -- "
                f"this driver stages into a scratch tree (Kaggle), and {linkname} "
                f"already holds real data")
        linkname.unlink()
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


def train_big(scene: str, dense: Path, run_dir: Path, max_iters: int, gpu: int | None,
              logf: Path, model: str = "splatfacto-big"):
    step = big_ckpt_step(run_dir)
    if step is not None and step >= max_iters - 1:
        print(f"[{scene}] backbone present (step {step}); skip training.", flush=True)
        return
    if run_dir.exists():
        shutil.rmtree(run_dir)
    env = {**os.environ}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = ["ns-train", model, "--data", str(dense),
           "--output-dir", str(run_dir), "--max-num-iterations", str(max_iters),
           "--viewer.quit-on-train-completion", "True",
           "--pipeline.model.rasterize-mode", "antialiased",
           "colmap", "--eval-mode", "all", "--colmap-path", "sparse/0",
           # Force full-res: nerfstudio's colmap dataparser auto-downscales any
           # image whose long side > 1600 px and then INTERACTIVELY prompts to
           # generate the downscaled copies -- in a non-interactive subprocess
           # that prompt hits EOFError and ns-train dies in seconds. Only bonsai
           # (1920x1080) exceeds 1600 (drones 1320, chair 1280), so it crashed on
           # both Kaggle and the rented box. downscale-factor 1 disables the
           # prompt AND keeps training at the full res our test poses + DIBR use.
           "--downscale-factor", "1"]
    print(f"[{scene}] + {' '.join(cmd)}", flush=True)
    logf.parent.mkdir(parents=True, exist_ok=True)
    with open(logf, "w") as f:
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    assert r.returncode == 0, f"[{scene}] ns-train failed -- see {logf}"


def refiner_v2(scene: str, run_dir: Path, max_pairs: int, iters: int, gpu: int | None,
               logf: Path, suffix: str = "_v2", blocks: str = "conv",
               evidence: bool = False, adv: float = 0.0, adv_warmup: int = 1000,
               init_from: Path | None = None, bs: int = 4, stage: str = "all"):
    env = {**os.environ}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [PY, str(REPO / "Analysis/10_refiner_pilot.py"), "--scene", scene,
           "--config", str(run_dir), "--ss", "2", "--sample", "cubic",
           "--base", "48", "--bs", str(bs), "--iters", str(iters), "--ema", "0.999",
           "--tta", "--suffix", suffix, "--max-pairs", str(max_pairs),
           "--blocks", blocks, "--stage", stage]
    if evidence:
        cmd.append("--evidence")
    if adv > 0:
        cmd += ["--adv", str(adv), "--adv-warmup", str(adv_warmup)]
    if init_from is not None:
        cmd += ["--init-from", str(init_from)]
    if stage == "all":
        cmd.append("--png")   # lossless test renders only on the shipping pass
    print(f"[{scene}] + {' '.join(cmd)}", flush=True)
    logf.parent.mkdir(parents=True, exist_ok=True)
    with open(logf, "w") as f:
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    assert r.returncode == 0, f"[{scene}] refiner failed -- see {logf}"


def refiner_stack(scene: str, run_dir: Path, max_pairs: int, iters: int, gpu: int | None,
                  log_root: Path, suffix: str, blocks: str, evidence: bool,
                  adv: float, adv_warmup: int, bs: int) -> Path:
    """Run the refiner and return the log the val_loss should be read from.

    With --adv the run is TWO stages, because that is the only configuration the
    W3 panel actually measured: a critic attached to a randomly-initialised net
    just trains the critic, so the adversarial arm was warm-started from the
    fully-trained regression arm's weights. Reproducing the +0.0064 (bonsai) /
    +0.0029 (chair) hold-out deltas therefore requires the same two stages here.

    Stage 1 (`--stage train`) skips apply_test entirely -- its renders are thrown
    away, only `refiner<suffix>_warm.pt` matters -- so the extra cost is the 6k
    training iters, NOT a second full DIBR pass: both stages share the same warp
    variant (`_ss2_cub_ev`), so stage 2 reads stage 1's pair/input cache.
    """
    if adv <= 0:
        rlog = log_root / scene / "refiner.log"
        refiner_v2(scene, run_dir, max_pairs, iters, gpu, rlog, suffix=suffix,
                   blocks=blocks, evidence=evidence, bs=bs)
        return rlog

    warm_suffix = f"{suffix}_warm"
    warm_ckpt = REFINER_OUT / scene / f"refiner{warm_suffix}.pt"
    wlog = log_root / scene / "refiner_warm.log"
    if warm_ckpt.exists():
        print(f"[{scene}] warm-start ckpt present ({warm_ckpt.name}); skip stage 1.", flush=True)
    else:
        print(f"[{scene}] refiner stage 1/2: regression warm start ({blocks}"
              f"{'+evidence' if evidence else ''})", flush=True)
        refiner_v2(scene, run_dir, max_pairs, iters, gpu, wlog, suffix=warm_suffix,
                   blocks=blocks, evidence=evidence, bs=bs, stage="train")
        assert warm_ckpt.exists(), f"[{scene}] stage 1 produced no {warm_ckpt} -- see {wlog}"

    rlog = log_root / scene / "refiner.log"
    print(f"[{scene}] refiner stage 2/2: adversarial (adv={adv}) from {warm_ckpt.name}",
          flush=True)
    refiner_v2(scene, run_dir, max_pairs, iters, gpu, rlog, suffix=suffix,
               blocks=blocks, evidence=evidence, adv=adv, adv_warmup=adv_warmup,
               init_from=warm_ckpt, bs=bs)
    return rlog


def collect(scene: str, out_root: Path, logf: Path, suffix: str = "_v2"):
    dst = out_root / scene
    dst.mkdir(parents=True, exist_ok=True)
    renders = REFINER_OUT / scene / f"renders_refined{suffix}"
    pngs = sorted(renders.glob("*.png"))
    assert pngs, f"[{scene}] no refined PNGs at {renders}"
    rdst = dst / f"renders_refined{suffix}"
    if rdst.exists():
        shutil.rmtree(rdst)
    shutil.copytree(renders, rdst)
    ckpt = REFINER_OUT / scene / f"refiner{suffix}.pt"
    if ckpt.exists():
        shutil.copy2(ckpt, dst / f"refiner{suffix}.pt")
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
            keep_intermediates: bool = False, model: str = "splatfacto-big",
            suffix: str = "_v2", blocks: str = "conv", evidence: bool = False,
            adv: float = 0.0, adv_warmup: int = 1000, bs: int = 4):
    done = out_root / scene / f"renders_refined{suffix}"
    if done.exists() and list(done.glob("*.png")):
        print(f"[{scene}] already collected; skip.", flush=True)
        return
    raw, dense = stage(scene, ds)
    run_dir = big_root / scene
    train_big(scene, dense, run_dir, max_iters, gpu, log_root / scene / "ns_train.log", model)
    rlog = refiner_stack(scene, run_dir, max_pairs, refiner_iters, gpu, log_root,
                         suffix, blocks, evidence, adv, adv_warmup, bs)
    collect(scene, out_root, rlog, suffix)
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
    ap.add_argument("--phase", choices=sorted(PHASES), default="phase1",
                    help="which raw/processed tree stage() rebuilds into; "
                         "round-2 scenes MUST use round2 (see PHASES)")
    ap.add_argument("--keep-intermediates", action="store_true",
                    help="do NOT purge big run + pair caches after each scene "
                         "(default purges to survive Kaggle's ~20 GB disk)")
    ap.add_argument("--base-scenes", nargs="*", default=[],
                    help="scenes to train with plain splatfacto instead of "
                         "splatfacto-big (merged with DEFAULT_BASE_SCENES; the "
                         "-big model OOMs 1920x1080 scenes like bonsai on a T4)")
    ap.add_argument("--big-scenes", nargs="*", default=[],
                    help="scenes to FORCE onto splatfacto-big, overriding "
                         "DEFAULT_BASE_SCENES/--base-scenes (e.g. bonsai on a "
                         "rented 48 GB card where -big fits)")
    # --- refiner architecture / loss (the W1+W3 indoor-validated stack) ---
    ap.add_argument("--suffix", default="_v2",
                    help="output tag: renders_refined<suffix>/ + refiner<suffix>.pt. "
                         "Use a NEW tag (e.g. _v3) for the adversarial stack so the "
                         "shipped _v2 renders stay intact as a per-scene fallback")
    ap.add_argument("--blocks", choices=["conv", "naf"], default="conv",
                    help="refiner block type; 'naf' is the W1-adopted NAFNet variant")
    ap.add_argument("--evidence", action="store_true",
                    help="20-channel evidence stack (per-neighbour warps + confidence "
                         "+ depth); pairs with --blocks naf. NOTE: ~3x the pair-cache "
                         "disk of the 7ch default -- see the notebook's disk guard")
    ap.add_argument("--adv", type=float, default=0.0,
                    help="PatchGAN weight. >0 makes the refiner a TWO-STAGE run "
                         "(regression warm start, then critic). 0.003 is the W3-adopted "
                         "value; 0.01 lost to it on both indoor scenes")
    ap.add_argument("--adv-warmup", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=4,
                    help="refiner batch size; the indoor W1/W3 panel used 2 (the 20ch "
                         "evidence stack is ~3x the activation memory of 7ch)")
    args = ap.parse_args()

    global RAW_PRIV, PROC
    RAW_PRIV, PROC = PHASES[args.phase]

    ds = Path(args.dataset) / "data"
    assert ds.is_dir(), f"{ds} not found (expected <dataset>/data/<scene>/...)"
    out_root = Path(args.out)
    # For round 2 the big run must land where 04_x3_dibr_pilot.find_config()
    # discovers it: runs/round2/phase_locked/<scene>.
    big_root = (REPO / "runs/round2/phase_locked" if args.phase == "round2"
                else Path("/kaggle/working/exp034_big"))
    log_root = Path(f"/kaggle/working/{args.phase}_logs")

    if args.parallel == "auto":
        try:
            import torch
            workers = max(1, torch.cuda.device_count())
        except Exception:
            workers = 1
    else:
        workers = int(args.parallel)

    base_scenes = (set(args.base_scenes) | DEFAULT_BASE_SCENES) - set(args.big_scenes)
    backbone = {s: ("splatfacto" if s in base_scenes else "splatfacto-big")
                for s in args.scenes}

    print(f"exp034 fleet: {len(args.scenes)} scenes, {workers} worker(s)", flush=True)
    print("  backbone: " + ", ".join(f"{s}={m}" for s, m in backbone.items()),
          flush=True)
    print(f"  refiner : suffix={args.suffix} blocks={args.blocks} "
          f"evidence={args.evidence} adv={args.adv} bs={args.bs} "
          f"({'TWO-STAGE warm+critic' if args.adv > 0 else 'single-stage regression'})",
          flush=True)
    # Per-scene failures MUST be isolated: one scene's ns-train/refiner crash
    # cannot be allowed to abort or hide its siblings. (The old `list(ex.map())`
    # re-raised the FIRST future in submission order, so a fail-fast scene 0
    # swallowed every later scene's result -- a finished sibling was lost with
    # no error shown.) Each scene records its own exception; we report a
    # DONE/FAILED roster at the end and only exit non-zero if EVERY scene failed.
    failures: dict[str, str] = {}

    def run_one(i: int, s: str):
        try:
            process(s, ds, out_root, args.max_iters, args.max_pairs,
                    args.refiner_iters, (i % workers) if workers > 1 else None,
                    big_root, log_root, args.keep_intermediates, backbone[s],
                    args.suffix, args.blocks, args.evidence, args.adv,
                    args.adv_warmup, args.bs)
        except Exception as e:  # noqa: BLE001 -- isolate; roster reports it
            failures[s] = f"{type(e).__name__}: {e}"
            print(f"[{s}] FAILED -- {failures[s]}", flush=True)

    if workers <= 1:
        for i, s in enumerate(args.scenes):
            run_one(i, s)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(lambda i_s: run_one(*i_s), list(enumerate(args.scenes))))

    print("\nfleet complete. collected scenes:", flush=True)
    for s in args.scenes:
        st = out_root / s / "STATUS.txt"
        if st.exists():
            print("  " + st.read_text().strip().replace("\n", " | "))
        else:
            print(f"  {s}: MISSING" + (f" ({failures[s]})" if s in failures else ""))
    if failures:
        print(f"\n{len(failures)}/{len(args.scenes)} scenes FAILED: "
              f"{sorted(failures)} -- see their ns_train.log / refiner.log; "
              f"succeeded scenes are collected and safe to package.", flush=True)
    # non-zero ONLY if nothing succeeded, so the package cell still captures
    # partial results (and re-runs skip already-collected scenes).
    if failures and len(failures) == len(args.scenes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
