"""Build a pinhole (undistorted) COLMAP scene for SSS from a competition
SIMPLE_RADIAL train scene, mirroring nerfstudio's internal cv2.undistort
(K unchanged) so the SSS backend sees the same data splatfacto did.

Usage: python make_undistorted_scene.py --src .../hcm0034/train --dst data/hcm0034_undist
"""
import argparse
import shutil
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

SSS = Path(__file__).parent / "3D-student-splatting-and-scooping"
sys.path.insert(0, str(SSS))
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary  # noqa: E402


def write_pinhole_cameras_bin(path: Path, cams: dict):
    """cams: {camera_id: (width, height, fx, fy, cx, cy)}"""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cid, (w, h, fx, fy, cx, cy) in cams.items():
            f.write(struct.pack("<iiQQ", cid, 1, w, h))  # model 1 = PINHOLE
            f.write(struct.pack("<dddd", fx, fy, cx, cy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--src-sparse", type=Path, default=None,
                    help="default <src>/sparse/0 (kaggle staging uses <src>/sparse0)")
    ap.add_argument("--src-images", type=Path, default=None,
                    help="default <src>/images")
    args = ap.parse_args()

    src_sparse = args.src_sparse or args.src / "sparse/0"
    src_images = args.src_images or args.src / "images"
    if not src_sparse.exists() and (args.src / "sparse0").exists():
        src_sparse = args.src / "sparse0"
    if not src_images.exists() and (args.src / "train_images").exists():
        src_images = args.src / "train_images"
    dst_sparse = args.dst / "sparse/0"
    dst_images = args.dst / "images"
    dst_sparse.mkdir(parents=True, exist_ok=True)
    dst_images.mkdir(parents=True, exist_ok=True)

    intr = read_intrinsics_binary(src_sparse / "cameras.bin")
    out_cams = {}
    maps = {}
    for cid, cam in intr.items():
        # Phase-1 drone scenes are SIMPLE_RADIAL; the round-2 indoor scenes
        # (bonsai, chair) are SIMPLE_PINHOLE and carry no distortion term.
        # k=0 makes the undistort an identity, which is the correct no-op.
        if cam.model == "SIMPLE_RADIAL":
            f_, cx, cy, k = cam.params
        elif cam.model == "SIMPLE_PINHOLE":
            f_, cx, cy = cam.params
            k = 0.0
        else:
            raise SystemExit(f"unsupported camera model {cam.model}")
        K = np.array([[f_, 0, cx], [0, f_, cy], [0, 0, 1]])
        dist = np.array([k, 0, 0, 0])
        maps[cid] = (K, dist)
        out_cams[cid] = (cam.width, cam.height, f_, f_, cx, cy)
    write_pinhole_cameras_bin(dst_sparse / "cameras.bin", out_cams)

    shutil.copy2(src_sparse / "points3D.bin", dst_sparse / "points3D.bin")

    # images.bin registers train AND test frames; keep only frames whose
    # photo exists in train/images (test views must not enter training).
    have = {p.name for p in src_images.iterdir()}
    extr = read_extrinsics_binary(src_sparse / "images.bin")
    kept = {iid: im for iid, im in extr.items() if im.name in have}
    print(f"images.bin: keeping {len(kept)}/{len(extr)} registered frames")
    with open(dst_sparse / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(kept)))
        for iid, im in kept.items():
            f.write(struct.pack("<i", iid))
            f.write(struct.pack("<dddd", *im.qvec))
            f.write(struct.pack("<ddd", *im.tvec))
            f.write(struct.pack("<i", im.camera_id))
            f.write(im.name.encode() + b"\x00")
            f.write(struct.pack("<Q", len(im.point3D_ids)))
            for (x, y), pid in zip(im.xys, im.point3D_ids):
                f.write(struct.pack("<ddq", x, y, pid))

    K, dist = next(iter(maps.values()))  # single shared camera in these scenes
    # k==0 (SIMPLE_PINHOLE): undistort is an identity transform, but running it
    # anyway would decode and RE-ENCODE every JPEG, costing real quality for no
    # geometric change. SSS must see byte-identical pixels to what splatfacto
    # reads, or the backend A/B is confounded by a lossy round-trip.
    identity = float(dist[0]) == 0.0
    if identity:
        print("k=0 -> pinhole passthrough, copying images without re-encode")
    srcs = sorted(src_images.iterdir())
    for i, p in enumerate(srcs):
        if identity:
            shutil.copy2(p, dst_images / p.name)
        else:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            und = cv2.undistort(img, K, dist)
            cv2.imwrite(str(dst_images / p.name), und, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if (i + 1) % 60 == 0:
            print(f"{i+1}/{len(srcs)}")
    print("done:", args.dst)


if __name__ == "__main__":
    main()
