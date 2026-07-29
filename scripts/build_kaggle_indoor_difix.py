"""Build the indoor Difix-LoRA-revival Kaggle upload zip (§3.5 of the top-1 plan).

One combined zip for BOTH indoor scenes -- the LoRA pools chair+bonsai render->GT
pairs and is hold-out gated on the 25 frames/scene. Payload = the same
train-minus-holdout backbones W9 used (so pairs are rendered through the
Kaggle-proven Warper + fix_paths), plus the raw GT, plus the W9 refiner hold-out
outputs (`renders_val_w9_a003`) so the notebook can also test Difix stacked ON TOP
of the shipped refiner output on the same 25 frames.

Zip layout (kaggle_upload/):
  code/bts-nvs/{src/**, Analysis/04_x3_dibr_pilot.py}
  data/raw/VAI_NVS_DATA_ROUND2/<scene>/train/{images/**, sparse/0/{cameras,images}.bin}
  data/processed/round2/<scene>/train_staging_holdout/{images/**, sparse/0/*.bin}
  data/processed/round2/<scene>/splits/val_ids.txt
  runs/round2/val_holdout/<scene>/train_staging_holdout/splatfacto/<ts>/
      {config.yml, dataparser_transforms.json, nerfstudio_models/step-*.ckpt}
  refiner_holdout/<scene>/renders_val_w9_a003/**   (25 refiner outputs on the hold-out)

The notebook copies code/bts-nvs -> REPO and symlinks REPO/data, REPO/runs into the
read-only mount so the Warper's REPO-relative reads + fix_paths resolve exactly as W9.

Run: conda run -n airace python scripts/build_kaggle_indoor_difix.py
"""
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = REPO.parent / "kaggle"

RAW = REPO / "data/raw/VAI_NVS_DATA_ROUND2"
PROC = REPO / "data/processed/round2"
RUNS = REPO / "runs/round2/val_holdout"
X5 = REPO / "Analysis/X5_refiner"

SCENES = ["chair", "bonsai"]
CODE_ITEMS = ["src", "Analysis/04_x3_dibr_pilot.py"]
REFINER_HOLDOUT = "renders_val_w9_a003"   # W9's shipped-weight (0.003) refiner hold-out output


def cp(src: Path, dst: Path):
    assert src.exists(), f"MISSING: {src}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(src, dst)


def find_backbone(scene: str) -> Path:
    base = RUNS / scene / "train_staging_holdout" / "splatfacto"
    runs = sorted(p for p in base.glob("*") if (p / "config.yml").exists())
    assert runs, f"{scene}: no hold-out backbone under {base}"
    bb = runs[-1]
    assert list((bb / "nerfstudio_models").glob("step-*.ckpt")), f"{scene}: no ckpt in {bb}"
    return bb


def main():
    KAGGLE.mkdir(exist_ok=True)
    root = KAGGLE / ".build_indoor_difix"
    stage = root / "kaggle_upload"
    if root.exists():
        shutil.rmtree(root)

    for item in CODE_ITEMS:
        cp(REPO / item, stage / "code/bts-nvs" / item)

    for scene in SCENES:
        cp(RAW / scene / "train/images",
           stage / f"data/raw/VAI_NVS_DATA_ROUND2/{scene}/train/images")
        for b in ("cameras.bin", "images.bin"):
            cp(RAW / scene / "train/sparse/0" / b,
               stage / f"data/raw/VAI_NVS_DATA_ROUND2/{scene}/train/sparse/0" / b)
        cp(PROC / scene / "train_staging_holdout",
           stage / f"data/processed/round2/{scene}/train_staging_holdout")
        cp(PROC / scene / "splits/val_ids.txt",
           stage / f"data/processed/round2/{scene}/splits/val_ids.txt")

        bb = find_backbone(scene)
        rel = bb.relative_to(REPO)
        for sub in ("config.yml", "dataparser_transforms.json"):
            cp(bb / sub, stage / rel / sub)
        for ck in (bb / "nerfstudio_models").glob("step-*.ckpt"):
            cp(ck, stage / rel / "nerfstudio_models" / ck.name)

        # W9 refiner hold-out output (the shipped-weight 0.003 arm) for the stacking gate
        cp(X5 / scene / REFINER_HOLDOUT, stage / f"refiner_holdout/{scene}/{REFINER_HOLDOUT}")

        # leak + integrity gate -- fail the build, not 12 h of T4 time
        val_ids = set((PROC / scene / "splits/val_ids.txt").read_text().split())
        staging = {p.name for p in (PROC / scene / "train_staging_holdout/images").iterdir()}
        raw_imgs = {p.name for p in (RAW / scene / "train/images").iterdir()}
        refined = {p.name for p in (X5 / scene / REFINER_HOLDOUT).iterdir()}
        assert len(val_ids) == 25, f"{scene}: {len(val_ids)} val ids"
        assert not (val_ids & staging), f"{scene}: LEAK {sorted(val_ids & staging)[:5]} in staging"
        assert val_ids <= raw_imgs, f"{scene}: missing GT {sorted(val_ids - raw_imgs)[:5]}"
        assert val_ids == refined, \
            f"{scene}: refiner hold-out set != val_ids (extra {sorted(refined - val_ids)[:3]}, " \
            f"missing {sorted(val_ids - refined)[:3]})"
        n_train = len(raw_imgs) - len(val_ids)
        print(f"  {scene}: {len(raw_imgs)} raw / {len(staging)} staging / {len(val_ids)} val "
              f"-> {n_train} train pairs + 25 hold-out | refiner outputs OK | backbone {rel.parts[-1]}")

    out_zip = KAGGLE / "kaggle-upload-indoor-difix.zip"
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(root))
    shutil.rmtree(root)
    print(f"\nbuilt {out_zip} ({out_zip.stat().st_size/1e9:.2f} GB)")
    print("Upload as a Kaggle dataset; set DATASET in kaggle/kaggle-round2-indoor-difix.ipynb.")


if __name__ == "__main__":
    main()
