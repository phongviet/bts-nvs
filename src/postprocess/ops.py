"""Post-process ops + encoders for exp011 (A.7) and the packaging path.

An op maps PIL RGB -> PIL RGB. An encoder controls how the image is written
to disk -- this matters because the scorer decodes our uploaded files, so
JPEG quantization is part of the pipeline whether we sweep it or not.
NOTE: src/render.py saves .JPG via PIL defaults = quality 75; encoders here
exist precisely to check how much that costs.

Names are stable identifiers used in CSVs and scene_overrides configs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

# ---------------- ops ----------------

def op_identity(img: Image.Image) -> Image.Image:
    return img


def make_unsharp(radius: float, percent: int) -> Callable[[Image.Image], Image.Image]:
    def op(img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=2))
    return op


def make_bilateral(d: int, sigma_color: int, sigma_space: int) -> Callable[[Image.Image], Image.Image]:
    def op(img: Image.Image) -> Image.Image:
        import cv2
        arr = cv2.bilateralFilter(np.asarray(img), d, sigma_color, sigma_space)
        return Image.fromarray(arr)
    return op


def make_nlmeans(h: int) -> Callable[[Image.Image], Image.Image]:
    def op(img: Image.Image) -> Image.Image:
        import cv2
        arr = cv2.fastNlMeansDenoisingColored(np.asarray(img), None, h, h, 7, 21)
        return Image.fromarray(arr)
    return op


OPS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "identity": op_identity,
    "unsharp_r1_p50": make_unsharp(1.0, 50),
    "unsharp_r1_p100": make_unsharp(1.0, 100),
    "unsharp_r2_p50": make_unsharp(2.0, 50),
    "unsharp_r2_p100": make_unsharp(2.0, 100),
    "bilateral_d5": make_bilateral(5, 30, 30),
    "nlmeans_h3": make_nlmeans(3),
}

# ---------------- encoders ----------------
# encoder name -> (save kwargs, forced suffix or None to keep the original name)

ENCODERS: dict[str, dict] = {
    "jpeg75": {"format": "JPEG", "quality": 75},   # PIL default -- what render.py ships today
    "jpeg90": {"format": "JPEG", "quality": 90},
    "jpeg95": {"format": "JPEG", "quality": 95},
    "jpeg98": {"format": "JPEG", "quality": 98},
    "jpeg95_444": {"format": "JPEG", "quality": 95, "subsampling": 0},
    "jpeg98_444": {"format": "JPEG", "quality": 98, "subsampling": 0},
    "png": {"format": "PNG"},
}


def process_image(src: Path, dst: Path, op_name: str, encoder_name: str):
    """Apply op, then encode. dst keeps src's filename (the submission requires
    exact image_name matches, so PNG data in a .JPG-named file is NOT allowed --
    encoder 'png' is only for local what-if scoring)."""
    img = Image.open(src).convert("RGB")
    img = OPS[op_name](img)
    kwargs = dict(ENCODERS[encoder_name])
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, **kwargs)


def process_dir(src_dir: Path, dst_dir: Path, op_name: str, encoder_name: str,
                exts=(".jpg", ".jpeg", ".png")) -> int:
    n = 0
    for p in sorted(src_dir.iterdir()):
        if p.suffix.lower() in exts:
            process_image(p, dst_dir / p.name, op_name, encoder_name)
            n += 1
    return n
