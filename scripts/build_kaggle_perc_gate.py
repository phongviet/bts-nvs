"""Build the Kaggle upload for the chair PERCEPTUAL-BACKBONE GATE (W11).

Why this run exists
-------------------
The perceptual backbone (`splatfacto-perceptual`, LPIPS(VGG) 0.1) is the only
backbone-side arm with a positive, reproduced signal on RAW hold-out renders:

    W9 hold-out backbone (plain splatfacto, antialiased)  raw Score 0.6506
    Kaggle E1  (splatfacto-perceptual, classic)           raw Score 0.6558  (+0.0052)
    Kaggle arm A (splatfacto-perceptual, classic)         raw Score 0.6561
    Kaggle arm D (plain splatfacto, classic, dense init)  raw Score 0.6500

Two independent Kaggle sessions put it at +0.005..+0.006, and `rasterize_mode` is worth only
~0.0006 on the raw render (0.6506 antialiased vs 0.6500 classic), so the raw gain is
real and not a rasterizer artefact.

What is NOT known is whether it survives the DIBR/refiner pipeline, because every
gate that tried to measure it on 2026-07-26 was void. Root cause (found 2026-07-26,
fixed in 04_x3_dibr_pilot.py): `fix_paths` inferred the backbone's training staging
from the run-tree shape (`cfg_path.parents[2].name`). `ns-train --output-dir X
--experiment-name chair` puts the SCENE in that slot, the lookup missed, and it fell
through to `train_staging_dense` -- a different, larger pose set (chair 205 vs 180).
That changes `dataparser_transform`/`dataparser_scale`, so every warp lands
misaligned, and on a hold-out backbone it re-admits the 25 val frames. Result: ~0.55
instead of ~0.67 through the refiner, on ANY backbone. The fix trusts the staging the
config records, and every run now prints `BACKBONE staging=... poses=...`.

The gate (four arms, three of them on shipped checkpoints)
---------------------------------------------------------
    anchor   W9 hold-out backbone (plain splatfacto, antialiased) -> MUST reproduce
             ctrl 0.6695 / adv0.003 0.6723. This is the harness check that was missing
             on 2026-07-26 and the proof that the fix above works.
    e1       local E1 perceptual backbone (classic), shipped as a checkpoint.
    perc_aa  splatfacto-perceptual + antialiased, TRAINED in-session -- the clean
             shipped-comparable arm (single variable vs `anchor`: the LPIPS term).

Verdict bar: perc_aa (or e1) must beat the anchor's adv0.003 arm by > +0.002 Score
AND on LPIPS. Anything less and the backbone family is closed.

Zip layout (kaggle_upload/):
  code/bts-nvs/{Analysis/04_x3_dibr_pilot.py, Analysis/10_refiner_pilot.py, src/**,
                docs/pip_freeze_week1.txt}
  data/raw/VAI_NVS_DATA_ROUND2/chair/train/{images/**, sparse/0/{cameras,images}.bin}
  data/processed/round2/chair/{train_staging_holdout/**, colmap_train_only/**,
                               splits/val_ids.txt}
  runs/round2/val_holdout/chair/train_staging_holdout/splatfacto/<ts>/**      (anchor)
  runs/round2/backbone_side/E1/chair/splatfacto-perceptual/run/**             (e1)

Both shipped backbones keep their ORIGINAL relative run paths on purpose: nerfstudio
reconstructs the checkpoint dir from `output_dir/experiment_name/method_name/timestamp`,
so moving them would break checkpoint discovery. The staging now resolves from the
config, not the path shape, so the differing shapes are harmless -- the notebook
asserts `staging=train_staging_holdout poses=180` from the log either way.

Run: conda run -n airace python scripts/build_kaggle_perc_gate.py
"""
import argparse
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = REPO.parent / "kaggle"

RAW = REPO / "data/raw/VAI_NVS_DATA_ROUND2"
PROC = REPO / "data/processed/round2"
SCENE = "chair"

# 10_refiner_pilot imports 04 by file; both import src. render_val.py + metrics.py live in src.
CODE_ITEMS = [
    "Analysis/04_x3_dibr_pilot.py",
    "Analysis/10_refiner_pilot.py",
    "src",
    "docs/pip_freeze_week1.txt",
]

# Shipped backbones: (arm tag, run dir relative to REPO).
ANCHOR_GLOB = "runs/round2/val_holdout/{scene}/train_staging_holdout/splatfacto/*"
E1_RUN = "runs/round2/backbone_side/E1/{scene}/splatfacto-perceptual/run"


def cp(src: Path, dst: Path):
    assert src.exists(), f"MISSING: {src}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(src, dst)


def ship_backbone(run: Path, stage: Path, tag: str):
    """Copy config + dataparser transforms + the final checkpoint, keeping the rel path."""
    assert (run / "config.yml").exists(), f"{tag}: no config.yml under {run}"
    ckpts = sorted((run / "nerfstudio_models").glob("step-*.ckpt"))
    assert ckpts, f"{tag}: {run} has no nerfstudio_models/step-*.ckpt"
    rel = run.relative_to(REPO)
    for sub in ("config.yml", "dataparser_transforms.json"):
        p = run / sub
        if p.exists():                      # transforms.json is absent on some older runs
            cp(p, stage / rel / sub)
    cp(ckpts[-1], stage / rel / "nerfstudio_models" / ckpts[-1].name)
    # The staging now resolves from the config, so assert the config actually names it.
    cfg = (run / "config.yml").read_text()
    assert "train_staging_holdout" in cfg, \
        f"{tag}: {run}/config.yml does not reference train_staging_holdout -- it was not " \
        f"trained on the hold-out split, so it would leak the 25 val frames"
    mb = ckpts[-1].stat().st_size / 1e6
    print(f"  {tag}: {rel} ({ckpts[-1].name}, {mb:.0f} MB)")
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--out", default=None, help="output zip path")
    args = ap.parse_args()
    scene = args.scene

    KAGGLE.mkdir(exist_ok=True)
    root = KAGGLE / f".build_perc_gate_{scene}"
    stage = root / "kaggle_upload"
    if root.exists():
        shutil.rmtree(root)

    print(f"[{scene}] staging ...")
    for item in CODE_ITEMS:
        cp(REPO / item, stage / "code/bts-nvs" / item)

    # The two code fixes this run depends on. A stale zip silently reproduces the void gate.
    dibr = (stage / "code/bts-nvs/Analysis/04_x3_dibr_pilot.py").read_text()
    assert "BACKBONE staging=" in dibr, "staged 04 lacks the staging-provenance print"
    assert 'stored = [getattr(config, "data", None)' in dibr, \
        "staged 04 still infers the staging from the run-tree shape (the 2026-07-26 bug)"
    assert ".owner" in dibr, "staged 04 lacks the depth-cache owner stamp"
    ref = (stage / "code/bts-nvs/Analysis/10_refiner_pilot.py").read_text()
    for tok, why in [("class PatchD", "PatchGAN critic"),
                     ('"--init-from"', "warm-start flag"),
                     ('ap.add_argument("--adv"', "--adv flag"),
                     ('ap.add_argument("--val-holdout"', "--val-holdout flag"),
                     ("d1.proj_in.weight", "naf checkpoint loader fix"),
                     ("_cut = None", "the _cut NameError fix (fresh apply dies without it)")]:
        assert tok in ref, f"staged refiner missing {why}"

    # raw GT for the held-out frames + the Warper's poses/intrinsics
    cp(RAW / scene / "train/images",
       stage / f"data/raw/VAI_NVS_DATA_ROUND2/{scene}/train/images")
    for b in ("cameras.bin", "images.bin"):
        cp(RAW / scene / "train/sparse/0" / b,
           stage / f"data/raw/VAI_NVS_DATA_ROUND2/{scene}/train/sparse/0" / b)

    # the backbone's datamanager data (train-minus-holdout), the split, and the pose source
    # render_val.py reads (colmap_train_only) for the raw hold-out score
    cp(PROC / scene / "train_staging_holdout",
       stage / f"data/processed/round2/{scene}/train_staging_holdout")
    cp(PROC / scene / "colmap_train_only",
       stage / f"data/processed/round2/{scene}/colmap_train_only")
    cp(PROC / scene / "splits/val_ids.txt",
       stage / f"data/processed/round2/{scene}/splits/val_ids.txt")

    print(f"[{scene}] shipped backbones:")
    anchors = sorted(p for p in REPO.glob(ANCHOR_GLOB.format(scene=scene))
                     if (p / "config.yml").exists())
    assert anchors, f"{scene}: no W9 hold-out backbone under {ANCHOR_GLOB}"
    ship_backbone(anchors[-1], stage, "anchor")
    ship_backbone(REPO / E1_RUN.format(scene=scene), stage, "e1")

    # leak + integrity gate -- fail the build, not six hours of T4 time
    val_ids = set((PROC / scene / "splits/val_ids.txt").read_text().split())
    staging_imgs = {p.name for p in (PROC / scene / "train_staging_holdout/images").iterdir()}
    raw_imgs = {p.name for p in (RAW / scene / "train/images").iterdir()}
    assert len(val_ids) == 25, f"{scene}: expected 25 val ids, got {len(val_ids)}"
    assert not (val_ids & staging_imgs), \
        f"{scene}: LEAK -- {sorted(val_ids & staging_imgs)[:5]} in backbone training staging"
    assert val_ids <= raw_imgs, \
        f"{scene}: missing GT for held-out {sorted(val_ids - raw_imgs)[:5]}"
    assert len(staging_imgs) == 180, \
        f"{scene}: hold-out staging has {len(staging_imgs)} images, expected 180 -- the " \
        f"180-vs-205 confusion is exactly what voided the 2026-07-26 gates"
    print(f"  {len(raw_imgs)} raw / {len(staging_imgs)} staging / {len(val_ids)} val, no leak")

    out_zip = Path(args.out) if args.out else KAGGLE / f"kaggle-upload-perc-gate-{scene}.zip"
    if out_zip.exists():
        out_zip.unlink()
    # STORED: JPGs, .bins and the ckpts do not compress; saves build + unzip time.
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(root))
    shutil.rmtree(root)
    print(f"[{scene}] built {out_zip} ({out_zip.stat().st_size/1e9:.2f} GB)")
    print(f"\nUpload as a Kaggle dataset, then set DATASET in "
          f"kaggle/kaggle-round2-{scene}-perc-gate.ipynb (T4 x2, Internet On).")


if __name__ == "__main__":
    main()
