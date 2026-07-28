"""Prepare a drone scene for the official-3DGS A/B: undistort to PINHOLE + cut a hold-out.

Why this exists
---------------
The drone scenes are SIMPLE_RADIAL (k1 ~ 0.0089). The official 3DGS COLMAP reader handles
only SIMPLE_PINHOLE and PINHOLE and asserts out on anything else, so the scene must be
undistorted before either arm can train on it.

That is not a workaround, it is what already happens: nerfstudio's FullImageDatamanager
undistorts every image before splatfacto ever sees it
(`full_images_datamanager.py:389 _undistort_image`, via
`cv2.getOptimalNewCameraMatrix(K, D, (W, H), 0)` + ROI crop). This script mirrors that
call exactly, so both arms train on the same pixels under the same intrinsics.

Also, unlike chair/bonsai, no drone scene has a hold-out split or a splatfacto anchor --
`runs/round2/val_holdout/` holds only the two indoor scenes. So the drone comparison is
self-contained: BOTH arms get trained in-session on the staging this script writes, and
the splatfacto side is a new anchor rather than the shipped number.

Output (data/processed/round2/<scene>/pinhole/):
  images_undist/**                     all 240 undistorted photos (JPEG q100)
  colmap_all/{cameras,images,points3D}.bin
      cameras: PINHOLE, intrinsics shifted by the ROI crop
      images:  poses copied verbatim -- undistortion does not move the camera
      points3D: the 2.0 M dense-init cloud (3D points are unaffected by undistortion),
                verified to share the world frame with colmap_train_only
  splits/{train_ids.txt,val_ids.txt,val_match.csv}   25 hold-outs, match-test mode

Run: conda run -n airace python scripts/prep_drone_pinhole.py --scene HCM0421
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RAW = REPO / "data/raw/VAI_NVS_DATA_ROUND2"
PROC = REPO / "data/processed/round2"

from nerfstudio.data.utils.colmap_parsing_utils import (  # noqa: E402
    Camera, read_cameras_binary, read_images_binary, write_cameras_binary,
    write_images_binary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="HCM0421")
    ap.add_argument("--n-val", type=int, default=25)
    ap.add_argument("--quality", type=int, default=100,
                    help="JPEG quality for the undistorted photos; these are the GT the "
                         "scores are computed against, so keep re-encoding loss minimal")
    args = ap.parse_args()
    scene = args.scene

    src = PROC / scene / "colmap_train_only"
    out = PROC / scene / "pinhole"
    imgs_out = out / "images_undist"
    imgs_out.mkdir(parents=True, exist_ok=True)

    cams = read_cameras_binary(src / "cameras.bin")
    assert len(cams) == 1, f"{scene}: expected a single shared camera, got {len(cams)}"
    cam = list(cams.values())[0]
    assert cam.model == "SIMPLE_RADIAL", \
        f"{scene}: camera model is {cam.model}; this script only handles SIMPLE_RADIAL"
    f, cx, cy, k1 = cam.params
    W, H = cam.width, cam.height
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64)

    # identical to nerfstudio's _undistort_image: alpha=0, then crop to the valid ROI
    newK, roi = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 0)
    x0, y0, w, h = roi
    print(f"[{scene}] {cam.model} {W}x{H} k1={k1:.5f} -> PINHOLE {w}x{h} "
          f"(roi x{x0} y{y0}), fx {newK[0,0]:.3f} fy {newK[1,1]:.3f}")

    ims = read_images_binary(src / "images.bin")
    raw_dir = RAW / scene / "train/images"
    for i, im in enumerate(sorted(ims.values(), key=lambda v: v.name), 1):
        dst = imgs_out / im.name
        if dst.exists():
            continue
        a = cv2.imread(str(raw_dir / im.name))
        assert a is not None, f"unreadable: {raw_dir / im.name}"
        assert (a.shape[1], a.shape[0]) == (W, H), \
            f"{im.name}: image is {a.shape[1]}x{a.shape[0]}, colmap says {W}x{H}"
        u = cv2.undistort(a, K, D, None, newK)[y0:y0 + h, x0:x0 + w]
        cv2.imwrite(str(dst), u, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        if i % 60 == 0:
            print(f"  undistorted {i}/{len(ims)}", flush=True)
    print(f"  {len(list(imgs_out.iterdir()))} undistorted -> {imgs_out}")

    # PINHOLE camera: the ROI crop moves the principal point, focal is unchanged
    cdir = out / "colmap_all"
    cdir.mkdir(parents=True, exist_ok=True)
    write_cameras_binary({1: Camera(id=1, model="PINHOLE", width=w, height=h,
                                    params=np.array([newK[0, 0], newK[1, 1],
                                                     newK[0, 2] - x0, newK[1, 2] - y0]))},
                         cdir / "cameras.bin")
    # extrinsics are untouched by undistortion; re-id the camera to the one written above
    write_images_binary({k: v._replace(camera_id=1) for k, v in ims.items()},
                        cdir / "images.bin")
    # 3D points are unaffected by undistortion. The dense-init cloud shares the world
    # frame with colmap_train_only (verified: identical tvecs on all 240 common images).
    shutil.copy2(PROC / scene / "colmap_dense_init/points3D.bin", cdir / "points3D.bin")

    # hold-out split, same mode and size as the indoor scenes
    splits = out / "splits"
    subprocess.run([sys.executable, str(REPO / "src/data_prep/make_val_split.py"),
                    "--mode", "match-test",
                    "--images-dir", str(imgs_out),
                    "--sparse-dir", str(cdir),
                    "--test-poses", str(RAW / scene / "test/test_poses.csv"),
                    "--out-dir", str(splits),
                    "--n-val", str(args.n_val)], check=True, cwd=REPO)

    val = sorted((splits / "val_ids.txt").read_text().split())
    train = sorted((splits / "train_ids.txt").read_text().split())
    assert len(val) == args.n_val, f"got {len(val)} val ids, expected {args.n_val}"
    assert not (set(val) & set(train)), "split overlaps"
    assert len(val) + len(train) == len(ims), \
        f"{len(train)}+{len(val)} != {len(ims)} registered images"
    print(f"[{scene}] {len(train)} train / {len(val)} hold-out -> {splits}")


if __name__ == "__main__":
    main()
