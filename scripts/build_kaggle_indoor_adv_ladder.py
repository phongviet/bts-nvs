"""Build the indoor adversarial-weight-ladder Kaggle upload zips (one per scene).

These datasets drive `kaggle/kaggle-round2-<scene>-adv-ladder.ipynb`, the Kaggle
port of the local W9 ladder (`scripts/run_W9_adv_ladder.sh`). The ladder is read
out on a GRADER-SHAPED 25-frame HOLD-OUT, never the LB (ruler doctrine: never
val_loss/LB for adversarial arms), so the payload must reproduce the exact
train-minus-holdout backbone the local ladder used -- we SHIP the already-trained
backbone rather than retrain on Kaggle, so the only variable across arms is --adv.

Zip layout (kaggle_upload/):
  code/bts-nvs/{Analysis/04_x3_dibr_pilot.py, Analysis/10_refiner_pilot.py,
                src/**, docs/pip_freeze_week1.txt}
  data/raw/VAI_NVS_DATA_ROUND2/<scene>/train/{images/**, sparse/0/{cameras,images}.bin}
  data/processed/round2/<scene>/train_staging_holdout/{images/**, sparse/0/*.bin}
  data/processed/round2/<scene>/splits/val_ids.txt
  runs/round2/val_holdout/<scene>/train_staging_holdout/splatfacto/<ts>/
      {config.yml, dataparser_transforms.json, nerfstudio_models/step-*.ckpt}

The notebook copies code/bts-nvs -> REPO and symlinks REPO/data, REPO/runs into
the read-only mount, so every REPO-relative path the refiner reads resolves.

Run: conda run -n airace python scripts/build_kaggle_indoor_adv_ladder.py --scenes chair bonsai
"""
import argparse
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = REPO.parent / "kaggle"

RAW = REPO / "data/raw/VAI_NVS_DATA_ROUND2"          # scene_raw() root for round-2
PROC = REPO / "data/processed/round2"
RUNS = REPO / "runs/round2/val_holdout"

# Code the refiner needs. 10_refiner_pilot imports 04 by file, and both import src.
CODE_ITEMS = [
    "Analysis/04_x3_dibr_pilot.py",
    "Analysis/10_refiner_pilot.py",
    "src",
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


def find_backbone(scene: str) -> Path:
    """The lone train_staging_holdout/splatfacto run dir (has config.yml + ckpt)."""
    base = RUNS / scene / "train_staging_holdout" / "splatfacto"
    runs = sorted(p for p in base.glob("*") if (p / "config.yml").exists())
    assert runs, f"{scene}: no hold-out backbone with config.yml under {base}"
    bb = runs[-1]
    ckpts = list((bb / "nerfstudio_models").glob("step-*.ckpt"))
    assert ckpts, f"{scene}: backbone {bb} has no nerfstudio_models/*.ckpt"
    return bb


def stage_scene(scene: str, stage: Path):
    # 1) code (once is enough, but idempotent copytree is cheap)
    for item in CODE_ITEMS:
        cp(REPO / item, stage / "code/bts-nvs" / item)

    # 2) raw train: GT for the held-out frames + the Warper's poses/intrinsics
    cp(RAW / scene / "train/images", stage / f"data/raw/VAI_NVS_DATA_ROUND2/{scene}/train/images")
    for b in ("cameras.bin", "images.bin"):
        cp(RAW / scene / "train/sparse/0" / b,
           stage / f"data/raw/VAI_NVS_DATA_ROUND2/{scene}/train/sparse/0" / b)

    # 3) processed hold-out staging: the backbone's datamanager data (train-minus-holdout)
    cp(PROC / scene / "train_staging_holdout",
       stage / f"data/processed/round2/{scene}/train_staging_holdout")
    cp(PROC / scene / "splits/val_ids.txt",
       stage / f"data/processed/round2/{scene}/splits/val_ids.txt")

    # 4) the trained hold-out backbone (config + transforms + ckpt)
    bb = find_backbone(scene)
    rel = bb.relative_to(REPO)          # runs/round2/val_holdout/<scene>/.../<ts>
    for sub in ("config.yml", "dataparser_transforms.json"):
        cp(bb / sub, stage / rel / sub)
    for ck in (bb / "nerfstudio_models").glob("step-*.ckpt"):
        cp(ck, stage / rel / "nerfstudio_models" / ck.name)

    # 5) leak + integrity gate -- fail the build, not an hour of T4 time
    val_ids = set((PROC / scene / "splits/val_ids.txt").read_text().split())
    staging_imgs = {p.name for p in (PROC / scene / "train_staging_holdout/images").iterdir()}
    raw_imgs = {p.name for p in (RAW / scene / "train/images").iterdir()}
    assert len(val_ids) == 25, f"{scene}: expected 25 val ids, got {len(val_ids)}"
    assert not (val_ids & staging_imgs), \
        f"{scene}: LEAK -- {sorted(val_ids & staging_imgs)[:5]} in backbone training staging"
    assert val_ids <= raw_imgs, \
        f"{scene}: missing GT for held-out frames {sorted(val_ids - raw_imgs)[:5]}"
    print(f"  {scene}: {len(raw_imgs)} raw / {len(staging_imgs)} staging / {len(val_ids)} val, "
          f"no leak | backbone {rel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["chair", "bonsai"])
    args = ap.parse_args()

    KAGGLE.mkdir(exist_ok=True)
    for scene in args.scenes:
        root = KAGGLE / f".build_indoor_adv_{scene}"
        stage = root / "kaggle_upload"
        if root.exists():
            shutil.rmtree(root)
        print(f"[{scene}] staging ...")
        stage_scene(scene, stage)

        out_zip = KAGGLE / f"kaggle-upload-indoor-adv-{scene}.zip"
        if out_zip.exists():
            out_zip.unlink()
        # STORED: JPGs + .bins + the ckpt do not compress; saves build/unzip time.
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as z:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(root))
        shutil.rmtree(root)
        print(f"[{scene}] built {out_zip} ({out_zip.stat().st_size/1e9:.2f} GB)\n")

    print("Upload each zip as its own Kaggle dataset; set DATASET in the matching "
          "kaggle-round2-<scene>-adv-ladder.ipynb.")


if __name__ == "__main__":
    main()
