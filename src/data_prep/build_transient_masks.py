"""Transient masks (exp008): mask people + vehicles out of the training loss.

The scenes have moving motorbikes/pedestrians/cars (see docs/strategy.md);
in the interpolation regime these are the main inconsistent-content source.
Same off-the-shelf ADE20K SegFormer as build_sky_masks.py (generic model, NOT
finetuned on BTS/telecom imagery -- provenance table already covers it), same
mask convention nerfstudio's ColmapDataParser expects: 255 = keep in loss,
0 = ignore (transient).

Because motorbikes/pedestrians are small at this GSD, masks are dilated a few
pixels (--dilate) to cover soft edges/shadows, and there is a --min-pixels
floor to ignore speck-sized false positives. SegFormer runs at 512x512, so a
motorbike in a 1320x989 nadir frame is only a few pixels to the model --
--tile-grid (default 2x3, 64px overlap) runs the model per tile and ORs the
results, which is what actually makes small street objects detectable
(verified on HCM0181 samples: 1x1 misses street motorbikes that 2x3 catches). QA: use --qa-dir to also write
side-by-side overlay JPEGs for visual inspection (~20 per pilot scene per the
plan).

Output: data/processed/phase1/<scene>/transient_masks/<image_stem>.png
Optionally symlinked into <staging>/masks (same mechanism as sky masks --
NOTE this replaces any existing masks symlink; sky masks were dropped in
Week 2 so that's the intended state).
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_CLASSES = ("person", "car", "minibike", "bicycle", "truck", "bus", "van")


def _tile_boxes(w: int, h: int, grid: tuple[int, int], overlap: int = 64):
    rows, cols = grid
    boxes = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, c * w // cols - overlap)
            y0 = max(0, r * h // rows - overlap)
            x1 = min(w, (c + 1) * w // cols + overlap)
            y1 = min(h, (r + 1) * h // rows + overlap)
            boxes.append((x0, y0, x1, y1))
    return boxes


def segment_transients(pipe, img: Image.Image, wanted: set[str], min_pixels: int,
                       grid: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    """Run segmentation per tile, OR the wanted-class masks into one bool map."""
    transient = np.zeros((img.height, img.width), dtype=bool)
    hits: list[str] = []
    for (x0, y0, x1, y1) in _tile_boxes(img.width, img.height, grid):
        for r in pipe(img.crop((x0, y0, x1, y1))):
            if r["label"] not in wanted:
                continue
            m = np.asarray(r["mask"]) > 0
            if r["label"] != "sky" and m.sum() < min_pixels:
                continue
            transient[y0:y1, x0:x1] |= m
            hits.append(r["label"])
    return transient, sorted(set(hits))


def build_transient_masks(images_dir: Path, out_dir: Path, classes: tuple[str, ...],
                          dilate: int, min_pixels: int, device: int = 0,
                          qa_dir: Path | None = None, qa_every: int = 12,
                          also_sky: bool = False, tile_grid: tuple[int, int] = (2, 3)):
    from scipy.ndimage import binary_dilation
    from transformers import pipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    if qa_dir is not None:
        qa_dir.mkdir(parents=True, exist_ok=True)
    pipe = pipeline("image-segmentation",
                    model="nvidia/segformer-b0-finetuned-ade-512-512", device=device)

    wanted = set(classes) | ({"sky"} if also_sky else set())
    image_paths = sorted(p for p in images_dir.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    n_masked = 0
    class_counts: dict[str, int] = {}
    for idx, p in enumerate(image_paths):
        img = Image.open(p).convert("RGB")
        transient, hits = segment_transients(pipe, img, wanted, min_pixels, tile_grid)
        for label in hits:
            class_counts[label] = class_counts.get(label, 0) + 1
        if dilate > 0 and transient.any():
            transient = binary_dilation(transient, iterations=dilate)

        mask = np.where(transient, 0, 255).astype(np.uint8)
        Image.fromarray(mask).save((out_dir / p.stem).with_suffix(".png"))
        if transient.any():
            n_masked += 1
        if qa_dir is not None and idx % qa_every == 0:
            overlay = np.asarray(img).copy()
            overlay[transient] = (0.4 * overlay[transient] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
            Image.fromarray(overlay).save(qa_dir / p.name, quality=85)
        print(f"{p.name}: {','.join(hits) if hits else 'clean'}")

    print(f"{images_dir}: {n_masked}/{len(image_paths)} images had transients masked -> {out_dir}")
    print(f"class hit counts (images): {class_counts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path,
                    help="e.g. data/raw/phase1/public_set/HCM0181")
    ap.add_argument("--processed-root", required=True, type=Path,
                    help="e.g. data/processed/phase1")
    ap.add_argument("--classes", default=";".join(DEFAULT_CLASSES),
                    help="semicolon-separated ADE20K labels to mask")
    ap.add_argument("--dilate", type=int, default=4, help="binary-dilation iterations (px)")
    ap.add_argument("--min-pixels", type=int, default=64,
                    help="ignore detections smaller than this (speck false positives)")
    ap.add_argument("--also-sky", action="store_true", help="merge sky into the mask too")
    ap.add_argument("--tile-grid", default="2x3",
                    help="RxC inference tiling; 1x1 = whole image (misses small objects)")
    ap.add_argument("--qa", action="store_true", help="write overlay JPEGs for visual QA")
    ap.add_argument("--qa-every", type=int, default=12, help="write every Nth overlay")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--staging-dir-name", default="train_staging_dense",
                    help="staging dir (under processed-root/<scene>/) to symlink masks/ into; "
                         "'none' to skip")
    args = ap.parse_args()

    scene = args.scene_dir.name
    scene_root = args.processed_root / scene
    masks_dir = scene_root / "transient_masks"
    qa_dir = scene_root / "transient_masks_qa" if args.qa else None
    rows, cols = (int(x) for x in args.tile_grid.lower().split("x"))
    build_transient_masks(args.scene_dir / "train" / "images", masks_dir,
                          tuple(args.classes.split(";")), args.dilate, args.min_pixels,
                          args.device, qa_dir, qa_every=args.qa_every,
                          also_sky=args.also_sky, tile_grid=(rows, cols))

    if args.staging_dir_name != "none":
        staging = scene_root / args.staging_dir_name
        if staging.exists():
            masks_link = staging / "masks"
            if masks_link.exists() or masks_link.is_symlink():
                masks_link.unlink()
            masks_link.symlink_to(masks_dir.resolve())
            print(f"Symlinked {masks_link} -> {masks_dir}")
        else:
            print(f"NOTE: {staging} missing -- symlink {masks_dir} as <staging>/masks manually "
                  f"or rerun after build_dense_colmap.py.")


if __name__ == "__main__":
    main()
