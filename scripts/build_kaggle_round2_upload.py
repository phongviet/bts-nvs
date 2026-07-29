"""Build the round-2 Kaggle upload zips. Two stages, one script.

  --stage init   -> kaggle-upload-round2-init.zip
                    Input for kaggle-dense-colmap-init.ipynb. That pipeline reads
                    <scene>/train/images + <scene>/colmap_train_only/*.bin and
                    emits colmap_dense_init/ + dense/fused_downsampled.ply.
                    Needed because the 5 round-2 drone scenes have NO dense init
                    yet, and dense-MVS init is worth +0.37 LB.

  --stage fleet  -> kaggle-upload-round2-fleet.zip
                    Input for the training fleet. Same allowlist shape as
                    build_kaggle_exp034_upload.py, but pointed at round-2 and
                    with dense_sparse0 required (asserts, so you cannot ship a
                    fleet zip that silently falls back to sparse init).

ALLOWLIST packaging, not rsync-excludes: the exp022 exclude bug (`data/` vs
`/data/`) cost a Kaggle session. Everything copied is asserted after the copy.

Run: conda run -n airace python scripts/build_kaggle_round2_upload.py --stage init
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = REPO.parent / "kaggle"

# The 5 drone scenes are the fleet's core job: 300 of 386 graded frames = 77.7%
# of the round-2 score. bonsai/chair (indoor, SIMPLE_PINHOLE) also have dense
# init + hold-out backbones and go in the FLEET zip too so a single upload can
# train any subset -- the operator picks scenes per Kaggle session via the
# driver's --scenes flag. INIT stays drone-only (indoor already has dense init).
DRONE = ["HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674"]
INDOOR = ["bonsai", "chair"]
FLEET = DRONE + INDOOR

RAW = REPO / "data/raw/round2/all"
PROC = REPO / "data/processed/round2"

# Code the fleet driver needs. Kept minimal and explicit.
CODE_ITEMS = [
    "src",
    "scripts/run_sweep.py",
    "Analysis/kaggle_exp034_fleet.py",  # the driver the notebook invokes
    "Analysis/04_x3_dibr_pilot.py",
    "Analysis/10_refiner_pilot.py",
    "Analysis/15_ladder_readout.py",
    "docs/pip_freeze_week1.txt",
]


def cp(src: Path, dst: Path):
    assert src.exists(), f"MISSING: {src}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(src, dst)


def stage_init(stage: Path):
    """Layout the dense-init pipeline expects: INPUT_ROOT/<scene>/..."""
    for s in DRONE:
        cp(RAW / s / "train/images", stage / f"round2/{s}/train/images")
        for b in ("cameras.bin", "images.bin", "points3D.bin"):
            cp(PROC / s / "colmap_train_only" / b,
               stage / f"round2/{s}/colmap_train_only" / b)
    for s in DRONE:
        for p in (f"round2/{s}/train/images/",
                  f"round2/{s}/colmap_train_only/points3D.bin"):
            assert (stage / p).exists(), f"stage missing {p}"
    # SCENES for the notebook is a list of paths relative to INPUT_ROOT.
    print("notebook config:")
    print(f'  INPUT_ROOT = ".../kaggle_upload/round2"')
    print(f"  SCENES = {DRONE}")
    print("  MAX_IMAGE_SIZE = 2000   # round-2 drone is 1320x989, never downsamples")


def stage_fleet(stage: Path, scenes: list[str] | None = None):
    scenes = scenes or FLEET
    for s in scenes:
        cp(RAW / s / "train/images", stage / f"data/{s}/train_images")
        # raw sparse: the Warper's poses/intrinsics source, and where the drone
        # SIMPLE_RADIAL k for the F1 remap is read from (test_poses.csv omits it;
        # indoor is SIMPLE_PINHOLE k=0, so its remap is identity but the model is
        # still carried so the driver reads intrinsics the same way everywhere).
        for b in ("cameras.bin", "images.bin"):
            cp(RAW / s / "train/sparse/0" / b, stage / f"data/{s}/raw_sparse0" / b)
        # dense init: REQUIRED. cp() asserts, so a missing dense init fails the
        # build instead of silently shipping a sparse-init fleet.
        for b in ("cameras.bin", "images.bin", "points3D.bin"):
            cp(PROC / s / "train_staging_dense/sparse/0" / b,
               stage / f"data/{s}/dense_sparse0" / b)
        cp(RAW / s / "test/test_poses.csv", stage / f"data/{s}/test_poses.csv")

    for item in CODE_ITEMS:
        cp(REPO / item, stage / "code/bts-nvs" / item)

    for s in scenes:
        for p in (f"data/{s}/train_images", f"data/{s}/raw_sparse0/cameras.bin",
                  f"data/{s}/dense_sparse0/points3D.bin", f"data/{s}/test_poses.csv"):
            assert (stage / p).exists(), f"stage missing {p}"
    print(f"fleet scenes ({len(scenes)}): {scenes}")
    print("  driver: python Analysis/kaggle_exp034_fleet.py --phase round2 "
          "--dataset /kaggle/input/<slug>/kaggle_upload --scenes <pick subset>")
    for p in ("code/bts-nvs/src/metrics.py",
              "code/bts-nvs/Analysis/kaggle_exp034_fleet.py",
              "code/bts-nvs/Analysis/10_refiner_pilot.py",
              "code/bts-nvs/Analysis/04_x3_dibr_pilot.py"):
        assert (stage / p).exists(), f"stage missing {p}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["init", "fleet"], required=True)
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="fleet stage only: restrict to a subset (e.g. a single "
                         "scene to re-run). The code payload -- incl. the driver "
                         "-- is always included, so a subset zip re-uploads the "
                         "current driver with minimal data. Default: all 7.")
    args = ap.parse_args()

    sel = None
    if args.stage == "fleet" and args.scenes:
        bad = [s for s in args.scenes if s not in FLEET]
        assert not bad, f"unknown fleet scene(s): {bad} (valid: {FLEET})"
        sel = args.scenes

    suffix = f"-{'_'.join(sel)}" if sel else ""
    root = KAGGLE / f"_stage_round2_{args.stage}{suffix}"
    stage = root / "kaggle_upload"
    out_zip = KAGGLE / f"kaggle-upload-round2-{args.stage}{suffix}.zip"

    if root.exists():
        shutil.rmtree(root)
    stage.mkdir(parents=True)

    if args.stage == "init":
        stage_init(stage)
    else:
        stage_fleet(stage, sel)

    scenes = DRONE if args.stage == "init" else (sel or FLEET)
    n_imgs = sum(1 for p in stage.rglob("*.JPG")) + sum(1 for p in stage.rglob("*.jpg"))
    print(f"staged: {len(scenes)} scenes, {n_imgs} train images")

    if out_zip.exists():
        out_zip.unlink()
    # STORED: JPGs and .bins do not compress; saves build AND unzip time on Kaggle.
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(root.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(root))
    print(f"built {out_zip} ({out_zip.stat().st_size/1e9:.2f} GB)")

    subprocess.run(["chmod", "-R", "u+w", str(root)], check=False)
    shutil.rmtree(root, ignore_errors=True)
    print("stage cleaned; upload as a Kaggle dataset.")


if __name__ == "__main__":
    main()
