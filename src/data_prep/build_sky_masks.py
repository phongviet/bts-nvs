"""Week 2 backend-locking: sky masks, so splatfacto's loss ignores sky pixels
during training (per docs/strategy.md: "Sky and thin metal are the two
BTS-specific failure modes. Sky generates floaters and wastes Gaussians").

Uses a generic, off-the-shelf ADE20K semantic segmentation model
(nvidia/segformer-b0-finetuned-ade-512-512, HuggingFace) to find the "sky"
class per train image -- NOT trained/finetuned on BTS/telecom imagery, see
docs/rules_and_constraints.md provenance table. Writes one binary PNG mask
per train image, same convention nerfstudio's ColmapDataParser expects
(`--masks-path`): 255 = keep pixel in the loss, 0 = ignore (sky).

Output: data/processed/phase1/<scene>/sky_masks/<image_stem>.png
Then symlinked into train_staging_dense/masks by make_train_dir's caller
(this script does the symlink itself, see main()).

Verified (2026-07-05, hcm0034 sample of 8 train images): sky is correctly
detected on images that show sky (BTS drone shots are often close/downward,
so many frames have none -- expected, not a bug).
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def build_sky_masks(images_dir: Path, out_dir: Path, device: int = 0):
    from transformers import pipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = pipeline("image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512", device=device)

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    n_with_sky = 0
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        results = pipe(img)
        sky_result = next((r for r in results if r["label"] == "sky"), None)

        mask = np.full((img.height, img.width), 255, dtype=np.uint8)
        if sky_result is not None:
            sky_mask = np.asarray(sky_result["mask"]) > 0
            mask[sky_mask] = 0
            n_with_sky += 1

        out_path = (out_dir / p.stem).with_suffix(".png")
        Image.fromarray(mask).save(out_path)
        print(f"{p.name}: {'sky masked' if sky_result is not None else 'no sky detected'}")

    print(f"{images_dir}: {n_with_sky}/{len(image_paths)} images had sky masked -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path, help="e.g. data/raw/phase1/public_set/hcm0034")
    ap.add_argument("--processed-root", required=True, type=Path, help="e.g. data/processed/phase1")
    ap.add_argument("--staging-dir-name", default="train_staging_dense",
                     help="which existing staging dir (under processed-root/<scene>/) to symlink masks/ into")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    scene = args.scene_dir.name
    scene_root = args.processed_root / scene
    masks_dir = scene_root / "sky_masks"
    build_sky_masks(args.scene_dir / "train" / "images", masks_dir, args.device)

    staging = scene_root / args.staging_dir_name
    if staging.exists():
        masks_link = staging / "masks"
        if masks_link.exists() or masks_link.is_symlink():
            masks_link.unlink()
        masks_link.symlink_to(masks_dir.resolve())
        print(f"Symlinked {masks_link} -> {masks_dir}")
    else:
        print(f"NOTE: {staging} does not exist yet -- run build_dense_colmap.py first, "
              f"then re-run this script (or manually symlink {masks_dir} as <staging>/masks).")


if __name__ == "__main__":
    main()
