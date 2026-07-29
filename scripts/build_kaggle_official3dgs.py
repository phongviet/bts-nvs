"""Build the Kaggle upload for the OFFICIAL 3DGS vs splatfacto raw A/B (chair).

Why this run exists
-------------------
Every backbone this project has measured is a nerfstudio/gsplat `splatfacto` variant.
The question here is different: how does the *reference implementation*
(graphdeco-inria/gaussian-splatting, CUDA rasterizer, its own densification schedule)
compare on the same scene, same training images, same hold-out, same scorer?

The measurement is RAW backbone quality only, directly against the number the W11 gate
established for the shipped backbone:

    anchor (splatfacto, antialiased, 30k)  raw hold-out Score 0.65062
      PSNR 23.765  SSIM 0.7688  LPIPS 0.3065

Comparability, and where it stops
---------------------------------
Identical: the 180 training images, the COLMAP world frame, the SfM point cloud used
for init, the 25 hold-out poses, the GT photos, and the scorer (`src/metrics.py`,
LPIPS vgg, PSNR_MAX 50).

NOT identical, on purpose -- these are what "different implementation" means and
removing them would defeat the point: the rasterizer (official EWA + 0.3 dilation vs
gsplat `antialiased`), the densification schedule, the LR schedule, SH warm-up.
This is a default-vs-default comparison at 30k iterations, not a controlled ablation
of any single knob.

No source patching is required
------------------------------
The two scene dirs share ONE world frame, verified: `colmap_train_only` and
`train_staging_holdout/sparse/0` give byte-identical tvecs for common images. Unlike
nerfstudio, the official `Scene` uses COLMAP poses directly with no dataparser
transform or scale, so poses need no conversion.

So the run is:
    train.py  -s chair/train   ->  180 cameras, all training (no --eval, no patch)
    render.py -m out -s chair/val --skip_test  ->  the 25 hold-out cameras, rendered
                                                   as that dir's "train" split
The `-s` override at render time is a supported `get_combined_args` path (command line
beats the model dir's stored cfg_args). The notebook asserts exactly 25 renders came
out, which is what fails loudly if the override is ever ignored.

Zip layout (kaggle_upload/):
  code/bts-nvs/src/**                       (metrics.py + utils, the shared scorer)
  chair/train/images/**                     180 hold-out training photos
  chair/train/sparse/0/{cameras,images,points3D}.bin
  chair/val/images/**                       the 25 hold-out GT photos (raw)
  chair/val/sparse/0/{cameras,images,points3D}.bin
  chair/val_ids.txt
  chair/anchor_raw_metrics.json             the splatfacto number being compared to

Run: conda run -n airace python scripts/build_kaggle_official3dgs.py
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = REPO.parent / "kaggle"
RAW = REPO / "data/raw/VAI_NVS_DATA_ROUND2"
PROC = REPO / "data/processed/round2"
SCENE = "chair"

# the splatfacto raw hold-out score this run is measured against
ANCHOR_METRICS = "runs/round2/val_holdout/{scene}/metrics_val_split.json"


def cp(src: Path, dst: Path):
    assert src.exists(), f"MISSING: {src}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(src, dst)


def write_val_colmap(src_bins: Path, out: Path, keep: set, cameras: Path, points: Path):
    """Write a COLMAP sparse/0 holding only the hold-out cameras.

    Poses are copied verbatim from `colmap_train_only`, which shares a world frame with
    the training reconstruction -- that is the whole reason no pose conversion is needed.
    """
    import sys
    sys.path.insert(0, str(REPO))
    from nerfstudio.data.utils.colmap_parsing_utils import (
        read_images_binary, write_images_binary)

    ims = read_images_binary(src_bins / "images.bin")
    sub = {k: v for k, v in ims.items() if v.name in keep}
    assert len(sub) == len(keep), \
        f"val colmap: matched {len(sub)} of {len(keep)} hold-out names in {src_bins}"
    out.mkdir(parents=True, exist_ok=True)
    write_images_binary(sub, out / "images.bin")
    cp(cameras, out / "cameras.bin")
    # readColmapSceneInfo always reads points3D to build its init ply, even when
    # render.py loads a trained checkpoint. The content is unused for rendering, so the
    # smaller train-only cloud goes here instead of the 72 MB dense one.
    cp(points, out / "points3D.bin")
    return len(sub)


def stage_scene(scene: str, stage: Path):
    """Stage one scene into the shared payload. Returns (n_train, n_val, has_anchor)."""
    # Drone scenes take the undistorted PINHOLE route (see prep_drone_pinhole.py) and
    # have no shipped anchor, so their splatfacto side is trained in-session too.
    drone = scene.upper().startswith("HCM")

    print(f"[{scene}] staging ...")
    if drone:
        pin = PROC / scene / "pinhole"
        assert pin.exists(), f"{scene}: run scripts/prep_drone_pinhole.py --scene {scene} first"
        val_ids = sorted((pin / "splits/val_ids.txt").read_text().split())
        train_ids = sorted((pin / "splits/train_ids.txt").read_text().split())
        img_src, sparse_src = pin / "images_undist", pin / "colmap_all"
        gt_src = img_src               # the undistorted photos are the GT
        n_train_expect = len(train_ids)
    else:
        val_ids = sorted((PROC / scene / "splits/val_ids.txt").read_text().split())
        hold = PROC / scene / "train_staging_holdout"
        train_ids = sorted(p.name for p in (hold / "images").iterdir())
        img_src, sparse_src = hold / "images", hold / "sparse/0"
        gt_src = RAW / scene / "train/images"
        n_train_expect = 180
    assert len(val_ids) == 25, f"expected 25 hold-out ids, got {len(val_ids)}"

    if drone:
        # train and val partition one undistorted image set, so nothing is duplicated
        for name in train_ids:
            cp(img_src / name, stage / f"{scene}/train/images" / name)
        n = write_val_colmap(sparse_src, stage / f"{scene}/train/sparse/0",
                             set(train_ids), sparse_src / "cameras.bin",
                             sparse_src / "points3D.bin")
        assert n == len(train_ids)
    else:
        cp(img_src, stage / f"{scene}/train/images")
        for b in ("cameras.bin", "images.bin", "points3D.bin"):
            cp(sparse_src / b, stage / f"{scene}/train/sparse/0" / b)

    train_imgs = {p.name for p in (stage / f"{scene}/train/images").iterdir()}
    assert len(train_imgs) == n_train_expect, \
        f"train dir has {len(train_imgs)} images, expected {n_train_expect}"
    assert not (set(val_ids) & train_imgs), \
        f"LEAK: {sorted(set(val_ids) & train_imgs)[:5]} present in the training images"

    print(f"[{scene}] hold-out scene ...")
    for name in val_ids:
        cp(gt_src / name, stage / f"{scene}/val/images" / name)
    # The hold-out poses cannot come from the training reconstruction -- it excludes them
    # by construction. They come from the all-image colmap, which shares its world frame
    # (verified for both regimes: identical tvecs on every common image).
    val_pose_src = sparse_src if drone else PROC / scene / "colmap_train_only"
    n = write_val_colmap(val_pose_src, stage / f"{scene}/val/sparse/0", set(val_ids),
                         sparse_src / "cameras.bin",
                         PROC / scene / "colmap_train_only/points3D.bin")
    (stage / f"{scene}/val_ids.txt").write_text("\n".join(val_ids) + "\n")

    anchor = REPO / ANCHOR_METRICS.format(scene=scene)
    if anchor.exists():
        cp(anchor, stage / f"{scene}/anchor_raw_metrics.json")
        a = json.loads(anchor.read_text())["mean"]
        print(f"  anchor (splatfacto) raw Score {a['score']:.5f}  "
              f"PSNR {a['psnr']:.3f} SSIM {a['ssim']:.4f} LPIPS {a['lpips']:.4f}")
    else:
        assert drone, f"{scene}: no anchor at {anchor} and it is not a drone scene"
        print("  no shipped anchor -- the notebook trains the splatfacto side in-session")
    print(f"  {len(train_imgs)} train / {n} hold-out cameras, no leak")
    return len(train_imgs), n, anchor.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["chair", "HCM0421"],
                    help="one indoor + one drone by default; they share one payload")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    KAGGLE.mkdir(exist_ok=True)
    root = KAGGLE / ".build_off3dgs"
    stage = root / "kaggle_upload"
    if root.exists():
        shutil.rmtree(root)

    # one copy of the scorer for every scene
    cp(REPO / "src", stage / "code/bts-nvs/src")
    manifest = {s: stage_scene(s, stage) for s in args.scenes}
    (stage / "scenes.json").write_text(json.dumps(
        {s: {"n_train": a, "n_val": b, "has_anchor": c} for s, (a, b, c) in manifest.items()},
        indent=2))

    out_zip = Path(args.out) if args.out else KAGGLE / "kaggle-upload-off3dgs.zip"
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(root))
    shutil.rmtree(root)
    print(f"\nbuilt {out_zip} ({out_zip.stat().st_size/1e6:.0f} MB) "
          f"covering {', '.join(args.scenes)}")
    print("Upload as ONE Kaggle dataset, then set DATASET in "
          "kaggle/kaggle-round2-official3dgs.ipynb (T4 x2, Internet On).")


if __name__ == "__main__":
    main()
