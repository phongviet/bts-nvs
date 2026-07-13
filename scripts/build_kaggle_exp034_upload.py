"""Build kaggle-upload-exp034.zip — ONE dataset serving both exp034 fleet
notebooks (A: HCM scenes, B: HNI scenes).

ALLOWLIST packaging (the exp022 rsync-exclude bug cost a Kaggle session; we
copy exactly what the notebooks need and assert every piece):

  kaggle_upload/
    code/bts-nvs/{src, scripts, configs, Analysis(04,10,15), docs/pip_freeze_week1.txt}
    data/<scene>/train_images/*.JPG              (raw train images)
    data/<scene>/raw_sparse0/{cameras,images}.bin (raw COLMAP: Warper poses/intrinsics)
    data/<scene>/dense_sparse0/{cameras,images,points3D}.bin (dense init for ns-train)
    data/<scene>/test_poses.csv

Run: conda run -n airace python scripts/build_kaggle_exp034_upload.py
"""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = REPO.parent / "kaggle"
STAGE = KAGGLE / "_stage_exp034/kaggle_upload"
OUT_ZIP = KAGGLE / "kaggle-upload-exp034.zip"

SCENES = ["HCM0249", "HCM0254", "HCM0276", "HCM1439",
          "HNI0131", "HNI0265", "HNI0366", "HNI0437"]
RAW = REPO / "data/raw/phase1/private_set1"
PROC = REPO / "data/processed/phase1"

CODE_ITEMS = [
    "src",
    "scripts/run_sweep.py",
    "configs/experiments/exp034_private_big_fleet.yaml",
    "Analysis/kaggle_exp034_fleet.py",   # the driver the notebook invokes (must be present!)
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


def main():
    if STAGE.parent.exists():
        shutil.rmtree(STAGE.parent)

    # --- code (allowlist) ---
    for item in CODE_ITEMS:
        cp(REPO / item, STAGE / "code/bts-nvs" / item)

    # --- per-scene data ---
    for s in SCENES:
        cp(RAW / s / "train/images", STAGE / f"data/{s}/train_images")
        for b in ("cameras.bin", "images.bin"):
            cp(RAW / s / "train/sparse/0" / b, STAGE / f"data/{s}/raw_sparse0" / b)
        for b in ("cameras.bin", "images.bin", "points3D.bin"):
            cp(PROC / s / "train_staging_dense/sparse/0" / b,
               STAGE / f"data/{s}/dense_sparse0" / b)
        cp(RAW / s / "test/test_poses.csv", STAGE / f"data/{s}/test_poses.csv")

    # --- post-copy asserts (the load-bearing files, verbatim paths the
    #     notebooks assert on) ---
    for s in SCENES:
        for p in (f"data/{s}/train_images", f"data/{s}/raw_sparse0/cameras.bin",
                  f"data/{s}/dense_sparse0/points3D.bin", f"data/{s}/test_poses.csv"):
            assert (STAGE / p).exists(), f"stage missing {p}"
    for p in ("code/bts-nvs/src/metrics.py", "code/bts-nvs/scripts/run_sweep.py",
              "code/bts-nvs/Analysis/kaggle_exp034_fleet.py",  # the driver the notebook runs
              "code/bts-nvs/Analysis/10_refiner_pilot.py",
              "code/bts-nvs/Analysis/04_x3_dibr_pilot.py",
              "code/bts-nvs/configs/experiments/exp034_private_big_fleet.yaml"):
        assert (STAGE / p).exists(), f"stage missing {p}"
    n_imgs = sum(1 for s in SCENES for _ in (STAGE / f"data/{s}/train_images").iterdir())
    print(f"staged: {len(SCENES)} scenes, {n_imgs} train images")

    # --- zip (STORED: JPGs/bins don't compress; faster build + unzip) ---
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(STAGE.parent.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(STAGE.parent))
    print(f"built {OUT_ZIP} ({OUT_ZIP.stat().st_size/1e9:.2f} GB)")
    subprocess.run(["chmod", "-R", "u+w", str(STAGE.parent)], check=False)  # copied JPGs are read-only
    shutil.rmtree(STAGE.parent, ignore_errors=True)
    print("stage cleaned; upload this zip as a Kaggle dataset (e.g. slug 'exp034-fleet').")


if __name__ == "__main__":
    main()
