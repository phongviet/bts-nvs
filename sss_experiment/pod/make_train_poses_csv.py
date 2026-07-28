"""Build a train_poses.csv (same schema as the competition test_poses.csv) from
an UNDISTORTED (pinhole) COLMAP scene's sparse/0. Lets render_test_csv.py (SSS)
and render_test_depth_csv.py (RaDe-GS) render/depth the TRAIN poses too, which
the COLMAP-native refiner needs as its training-pair render channel + depth_T.

Columns: image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height
(qvec,tvec are COLMAP world-to-camera, straight from images.bin.)

Run: python make_train_poses_csv.py --scene-dir <train_pinhole> --out <csv>
"""
import argparse
import csv
import struct
from collections import namedtuple
from pathlib import Path

_Cam = namedtuple("Cam", ["model", "width", "height", "params"])
_Img = namedtuple("Img", ["qvec", "tvec", "camera_id", "name"])
_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4)}


def read_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
            name, npar = _MODELS[mid]
            params = struct.unpack("<" + "d" * npar, f.read(8 * npar))
            cams[cid] = _Cam(name, w, h, params)
    return cams


def read_images(path):
    imgs = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            p = struct.unpack("<idddddddi", f.read(64))
            qvec, tvec, cid = p[1:5], p[5:8], p[8]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (npts,) = struct.unpack("<Q", f.read(8))
            f.read(24 * npts)
            imgs[p[0]] = _Img(qvec, tvec, cid, name.decode())
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path,
                    help="pinhole scene dir containing sparse/0/{cameras,images}.bin")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    sp = args.scene_dir / "sparse" / "0"
    cams = read_cameras(sp / "cameras.bin")
    imgs = read_images(sp / "images.bin")
    rows = []
    for im in imgs.values():
        c = cams[im.camera_id]
        if c.model == "SIMPLE_PINHOLE":
            f, cx, cy = c.params
            fx = fy = f
        elif c.model == "PINHOLE":
            fx, fy, cx, cy = c.params
        else:
            raise SystemExit(f"{im.name}: expected pinhole, got {c.model} "
                             f"(undistort first)")
        qw, qx, qy, qz = im.qvec
        tx, ty, tz = im.tvec
        rows.append(dict(image_name=im.name, qw=qw, qx=qx, qy=qy, qz=qz,
                         tx=tx, ty=ty, tz=tz, fx=fx, fy=fy, cx=cx, cy=cy,
                         width=c.width, height=c.height))
    rows.sort(key=lambda r: r["image_name"])
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} train poses -> {args.out} "
          f"({rows[0]['width']}x{rows[0]['height']}, fx={rows[0]['fx']:.1f})")


if __name__ == "__main__":
    main()
