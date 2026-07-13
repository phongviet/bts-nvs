"""Analysis 02 / Experiment X1: score the 'copy the nearest train image' baseline.

For each public-scene test pose, submit (unmodified) the train image whose
camera center is closest. This is the zero-effort floor of the image-based-
rendering family: it measures how much of the test view's content already
exists in a single nearby train photo, with NO geometric correction at all.

Interpretation guide:
  - X1 score ~ 0.5+ despite raw misalignment => test views are near-duplicates
    of train views; warping (X3) should dominate pure 3DGS reconstruction.
  - X1 LPIPS vs 3DGS LPIPS is the key column: LPIPS is fairly tolerant of small
    global misalignment but hates synthesis blur — if X1's LPIPS already rivals
    the 3DGS render's, real-pixel approaches win the 0.4-weighted term.

Run: conda run -n airace python Analysis/02_x1_nearest_copy.py
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "phase1" / "public_set"
OUT = Path(__file__).resolve().parent / "X1_nearest_copy"
sys.path.insert(0, str(REPO))

from src.metrics import compute_metrics  # noqa: E402
from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary  # noqa: E402


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def cam_center(q, t):
    return -qvec2rotmat(q).T @ np.asarray(t)


def main():
    summary = []
    for scene_dir in sorted(RAW.iterdir()):
        scene = scene_dir.name
        ims = read_images_binary(scene_dir / "train" / "sparse" / "0" / "images.bin")
        train_names = {p.name for p in (scene_dir / "train" / "images").iterdir()}
        train = [(im.name, cam_center(im.qvec, im.tvec)) for im in ims.values()
                 if im.name in train_names]
        centers = np.stack([c for _, c in train])

        render_dir = OUT / scene
        render_dir.mkdir(parents=True, exist_ok=True)
        with open(scene_dir / "test" / "test_poses.csv") as f:
            for r in csv.DictReader(f):
                C = cam_center([float(r[k]) for k in ("qw", "qx", "qy", "qz")],
                               [float(r[k]) for k in ("tx", "ty", "tz")])
                nn = train[int(np.argmin(np.linalg.norm(centers - C, axis=1)))][0]
                shutil.copyfile(scene_dir / "train" / "images" / nn,
                                render_dir / r["image_name"])

        res = compute_metrics(render_dir, scene_dir / "test" / "images",
                              lpips_net="vgg", psnr_max=50.0)
        m = res["mean"]
        (render_dir / "metrics.json").write_text(json.dumps(res, indent=2))
        print(f"{scene:9s} X1 nearest-copy: PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} "
              f"LPIPS={m['lpips']:.4f} Score={m['score']:.4f}")
        summary.append(dict(scene=scene, **{k: round(m[k], 4) for k in
                                            ("psnr", "ssim", "lpips", "score")}))

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"mean Score = {np.mean([s['score'] for s in summary]):.4f}")


if __name__ == "__main__":
    main()
