"""Analysis 24: build the ROUND-2 submission zip. Round-2 counterpart of
Analysis/14_build_v2_submission.py, which was hard-wired to the phase-1
public/private split and is now retired for submission use.

What changes from 14 (all of it forced by the round-2 data):

1. ONE graded set of 7 scenes, no public/private split. Every scene's bytes
   count against the same cap, so the encode allocation pools all 386 frames.

2. The byte budget is no longer binding at q95. Round-2 is 503.2 M pixels vs
   phase-1's 626.6 M (x0.803), and phase-1 measured q95 4:4:4 = 321.7 MB /
   q96 = 353 MB. Scaling: q95 ~ 258 MB, q98 ~ 312 MB -- q98 FITS under the cap
   that forced q95 in phase 1. Measured value of that step: ~+0.003 Score.
   So the knapsack FLOOR starts at 98 and steps down only on overrun, rather
   than starting at 95 and spending headroom upward. Floor semantics are
   unchanged: no image can land below the floor, so the step is non-negative
   by construction.

3. Mixed resolution and orientation. Drone 1320x989 landscape, bonsai
   1920x1080, chair 720x1280 PORTRAIT. Every frame is checked against the
   W,H declared in that scene's test_poses.csv before packaging -- a silent
   landscape assumption anywhere upstream dies here instead of on the LB.

The F1 distortion remap is NOT applied here; it belongs to the render chain
upstream (drone scenes only, k from cameras.bin -- bonsai/chair are
SIMPLE_PINHOLE k=0 and must be rendered without it). This script only asserts
the geometry it is handed.

Run: conda run -n airace python Analysis/24_build_round2_submission.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_by_path(mod: str, rel: str):
    spec = importlib.util.spec_from_file_location(mod, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ref10 = _load_by_path("ref10", "Analysis/10_refiner_pilot.py")
knap16 = _load_by_path("knap16", "Analysis/16_encode_knapsack.py")

X5 = REPO / "Analysis/X5_refiner"
X3 = REPO / "Analysis/X3_dibr"
RAW = REPO / "data/raw/round2/all"

# All 7 graded scenes. Per-image score weight, for reading the byte report:
# drone 300/386 = 77.7%, chair 58/386 = 15.0%, bonsai 28/386 = 7.3%.
DRONE = ["HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674"]
INDOOR = ["bonsai", "chair"]
SCENES = DRONE + INDOOR

# Budget is expressed in MiB (2**20), NOT decimal MB, because the upload page
# reports MiB -- a 340,007,763 B zip displays there as 324.3, and the old decimal
# BUDGET_MB = 340 was therefore leaving ~27 MB of the 350 MiB cap unspent.
#
# 348 MiB = 364,904,448 B = 364.9 decimal MB, i.e. 2 MiB of slack under the cap.
# That is what makes a **q98 floor affordable** (q98 over all 386 frames measured
# 357.1 decimal MB = 340.5 MiB, which overran the old 340 decimal budget and forced
# the knapsack down to a q97 floor).
#
# ASSUMPTION, stated because it is the one thing that can bite: this treats the
# competition's "350MB" as 350 MiB. Evidence is that the upload page displays MiB.
# If it turns out to be decimal, pass --budget-mib 333 (= 349.2 decimal MB) and the
# floor will step back down to q97 on its own. A rejected upload is recoverable.
BUDGET_MIB = 348


def expected_wh(scene: str) -> tuple[int, int]:
    """W,H every frame of this scene must have, read from the pose CSV itself."""
    csv_path = RAW / scene / "test/test_poses.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    wh = {(int(float(r["width"])), int(float(r["height"]))) for r in rows}
    assert len(wh) == 1, f"{scene}: mixed sizes in test_poses.csv: {wh}"
    return wh.pop()


def test_names(scene: str) -> list[str]:
    with open(RAW / scene / "test/test_poses.csv") as f:
        return [r["image_name"] for r in csv.DictReader(f)]


def png_renders(scene: str, suffix: str):
    """Lossless PNG renders (Kaggle fleet output or local --png apply).

    PNG stems are the test name minus its extension, and round-2 mixes
    extension CASE: drone frames are `.JPG`, bonsai/chair are `.jpg`. Restore
    the name from the pose CSV rather than appending a literal -- the phase-1
    builder hard-coded ".JPG", which would silently rename every indoor frame
    and fail packaging on 86 of 386 files.
    """
    d = X5 / scene / f"renders_refined{suffix}"
    pngs = sorted(d.glob("*.png")) if d.exists() else []
    if not pngs:
        return None
    by_stem = {Path(n).stem: n for n in test_names(scene)}
    out = []
    for f in pngs:
        name = by_stem.get(f.stem)
        assert name, f"{scene}: render {f.name} has no matching test pose"
        out.append((name, Image.open(f).convert("RGB")))
    return out


def apply_refiner(scene: str, suffix: str, variant: str, device):
    ckpt = X5 / scene / f"refiner{suffix}.pt"
    icache = X5 / scene / f"test_inputs{variant}"
    files = sorted(icache.glob("*.npz"))
    if not (ckpt.exists() and files):
        return None
    net = ref10.load_refiner(ckpt, device)
    out = []
    for fp in files:
        inp = np.load(fp)["inp"].astype(np.float32).transpose(2, 0, 1)
        img = ref10._net_apply(net, inp, device, tta=True)
        out.append((fp.name[:-4], Image.fromarray((img * 255).astype(np.uint8))))
    del net
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def fallback_images(scene: str):
    """Degrade one rung at a time so a partial fleet still builds a full zip."""
    for d in (X5 / scene / "renders_refined", X3 / scene / "renders_g0.18",
              REPO / f"runs/round2/phase_locked/{scene}/renders_test"):
        imgs = sorted(list(d.glob("*.JPG")) + list(d.glob("*.jpg"))) if d.exists() else []
        if imgs:
            print(f"  {scene}: FALLBACK source {d.relative_to(REPO)}")
            return [(f.name, Image.open(f).convert("RGB")) for f in imgs]
    raise FileNotFoundError(f"no render source for {scene}")


def validate(scene: str, imgs) -> None:
    """Names, count, and exact W x H against the pose CSV. Runs BEFORE encode."""
    want_names = test_names(scene)
    w, h = expected_wh(scene)
    got = {n for n, _ in imgs}
    missing = [n for n in want_names if n not in got]
    extra = sorted(got - set(want_names))
    assert not missing, f"{scene}: {len(missing)} frames missing, e.g. {missing[:3]}"
    assert not extra, f"{scene}: {len(extra)} unexpected frames, e.g. {extra[:3]}"
    bad = [(n, im.size) for n, im in imgs if im.size != (w, h)]
    assert not bad, (f"{scene}: expected {w}x{h}, got e.g. {bad[:3]} "
                     f"({'PORTRAIT scene' if h > w else 'landscape scene'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v2")
    ap.add_argument("--variant", default="_ss2_cub")
    ap.add_argument("--out", default="round2_v1_results")
    ap.add_argument("--qfloor", type=int, default=98,
                    help="knapsack floor quality; steps DOWN on budget overrun. "
                         "98 is the round-2 default because the pixel count "
                         "dropped 20%% vs phase 1 (see module docstring)")
    ap.add_argument("--qmax", type=int, default=100, help="knapsack ceiling")
    ap.add_argument("--budget-mib", type=int, default=BUDGET_MIB,
                    help="size budget in MiB (2**20) -- the unit the upload page "
                         "displays. Default 348 keeps a q98 floor affordable under a "
                         "350 MiB cap; use 333 if the cap turns out to be decimal MB")
    # DEFAULT OFF: measured WORSE (Jul-23). Scene-weighting maximises the
    # scene-mean of the knapsack's INNER fidelity proxy, but that proxy is
    # ms-ssim-tuned while the grader weights LPIPS 0.4 -- so weighting amplifies
    # the misspecification. Empirically v8 (weighted) lost 0.00039 scene-mean to
    # v7a (unweighted): bonsai's curve is SATURATED (+27.5% bytes bought +0.00012,
    # and its LPIPS got *worse*) while chair's is steep (-14.4% bytes cost 0.00282).
    # The greedy already allocates by marginal gain per byte, which is the right
    # criterion. Re-enable only after the inner metric is made grader-shaped.
    ap.add_argument("--scene-weighted", dest="scene_weighted",
                    action="store_true", default=False,
                    help="weight marginal gain by 1/(n_scenes*frames_in_scene); "
                         "MEASURED WORSE, see comment above")
    args = ap.parse_args()
    budget = args.budget_mib * 2**20

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = REPO / "submissions/round2" / args.out
    stage = out_root / "renders"
    if stage.exists():
        shutil.rmtree(stage)

    # 1) float-quality images per scene, best available rung
    per_scene, n_refined = {}, 0
    for s in SCENES:
        imgs = png_renders(s, args.suffix)
        if imgs is None:
            imgs = apply_refiner(s, args.suffix, args.variant, device)
        if imgs is None:
            imgs = apply_refiner(s, args.suffix, args.variant + "_bb", device)
        if imgs is not None:
            n_refined += 1
            print(f"  {s}: refined ({len(imgs)} views)")
        else:
            imgs = fallback_images(s)
        validate(s, imgs)
        per_scene[s] = imgs

    # 2) pooled knapsack over ALL 386 graded frames, floor stepping down on
    #    overrun. Pooling across scenes is what lets a flat drone sky frame
    #    fund a busy chair frame.
    pooled = [(f"{s}/{name}", im) for s in SCENES for name, im in per_scene[s]]

    # Scene-weighted marginal gain. The grader reports num_scenes/matched_scenes
    # and averages PER SCENE, so each scene owns 1/7 of the score regardless of
    # its frame count -- a 28-frame bonsai frame is worth 60/28 = 2.14x a drone
    # frame. The pooled knapsack used to weight every frame equally, which meant
    # the 300 drone frames (77.7% of frames but only 71.4% of the score) quietly
    # outbid indoor for the byte budget. Measured on the v7 build: q97->q98 moved
    # chair DOWN from q99/q100 while drones went up.
    # Normalised to mean 1.0 so the ratios stay on their old scale.
    n_scenes = len(SCENES)
    raw = {f"{s}/{name}": 1.0 / (n_scenes * len(per_scene[s]))
           for s in SCENES for name, _ in per_scene[s]}
    mean_w = sum(raw.values()) / len(raw)
    weights = {k: v / mean_w for k, v in raw.items()}
    if args.scene_weighted:
        print("  scene-weighted knapsack: " + ", ".join(
            f"{s}x{weights[f'{s}/{per_scene[s][0][0]}']:.2f}" for s in SCENES))
    else:
        weights = None
        print("  UNWEIGHTED knapsack (frame-mean, not scene-mean)")

    for floor in range(args.qfloor, 89, -1):
        alloc = knap16.allocate(pooled, budget,
                                qualities=range(floor, args.qmax + 1),
                                subsampling=0, tune="ms-ssim", weights=weights)
        total = sum(len(d) for d, _ in alloc.values())
        qs = sorted(q for _, q in alloc.values())
        print(f"  floor q{floor}: {total/2**20:.1f} MiB / {total/1e6:.1f} MB "
              f"(q{qs[0]}..q{qs[-1]})")
        if total <= budget:
            break
    else:
        raise SystemExit(f"cannot fit {args.budget_mib} MiB even at floor q90")
    print(f"  CHOSEN floor q{floor}: {total/2**20:.1f} MiB of {args.budget_mib} MiB "
          f"({total/1e6:.1f} MB decimal)")
    if floor < args.qfloor:
        print(f"  NOTE: stepped down from the requested q{args.qfloor} floor")

    enc = {s: {} for s in SCENES}
    for key, (data, _q) in alloc.items():
        s, name = key.split("/", 1)
        enc[s][name] = data

    # per-scene byte report, ordered by score weight
    for s in SCENES:
        mb = sum(len(b) for b in enc[s].values()) / 1e6
        n = len(enc[s])
        print(f"    {s:9s} {n:3d} frames  {mb:6.1f} MB  ({mb/n*1e3:.0f} KB/frame)")

    # 3) stage + package
    for s in SCENES:
        d = stage / s / "renders_test"
        d.mkdir(parents=True, exist_ok=True)
        for name, data in enc[s].items():
            (d / name).write_bytes(data)

    final = out_root / "submission_round2.zip"
    subprocess.run([sys.executable, str(REPO / "src/package_submission.py"),
                    "--runs-dir", str(stage), "--scenes", *SCENES,
                    "--poses-root", str(RAW), "--out", str(final)], check=True)

    print(f"\n{n_refined}/{len(SCENES)} scenes on the refiner. "
          f"{final} = {final.stat().st_size/1e6:.0f} MB @ floor q{floor} 4:4:4")


if __name__ == "__main__":
    main()
