"""Render images from a trained Nerfstudio checkpoint at arbitrary poses.

Two modes:
  --mode test   : render every pose in a competition test_poses.csv
  --mode train_check : re-render a handful of TRAIN images (from the scene's
                        COLMAP sparse/0) so you can diff against their GT --
                        this is the Week-1 Day-2 pose-convention sanity check.
                        If these don't match GT, the COLMAP->Nerfstudio
                        conversion below (or the dataparser transform) is wrong.

Pose convention notes (must hold for both modes to agree):
  - Input poses (test_poses.csv AND COLMAP images.bin) are qw,qx,qy,qz,tx,ty,tz
    in COLMAP's world-to-camera / OpenCV convention.
  - Nerfstudio's colmap dataparser converts to its own convention with:
      c2w = inverse(w2c); c2w[0:3,1:3] *= -1; c2w = c2w[[1,0,2,3]]; c2w[2,:] *= -1
    then applies a single global `dataparser_transform` (3x4) + `dataparser_scale`
    (computed once from the training poses, saved in the training run) to every
    pose, train AND test alike. We replicate both steps here using the values
    Nerfstudio itself produced for this checkpoint (loaded via eval_setup), so
    test poses land in the exact same space the model was trained in.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def colmap_pose_to_c2w(qvec, tvec) -> np.ndarray:
    """world-to-camera (qvec,tvec) -> nerfstudio-convention camera-to-world 4x4."""
    rotation = qvec2rotmat(qvec)
    translation = np.array(tvec).reshape(3, 1)
    w2c = np.eye(4)
    w2c[:3, :3] = rotation
    w2c[:3, 3:4] = translation
    c2w = np.linalg.inv(w2c)
    c2w[0:3, 1:3] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[2, :] *= -1
    return c2w


def apply_dataparser_transform(c2w: np.ndarray, transform: np.ndarray, scale: float) -> np.ndarray:
    """transform: (3,4) or (4,4) applied the same way nerfstudio's dataparser does."""
    c2w_h = np.eye(4)
    c2w_h[:3, :4] = c2w[:3, :4]
    if transform.shape[0] == 3:
        t = np.eye(4)
        t[:3, :4] = transform
        transform = t
    out = transform @ c2w_h
    out[:3, 3] *= scale
    return out[:3, :4]


def load_test_poses(csv_path: Path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "image_name": r["image_name"],
                "qvec": [float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])],
                "tvec": [float(r["tx"]), float(r["ty"]), float(r["tz"])],
                "fx": float(r["fx"]), "fy": float(r["fy"]),
                "cx": float(r["cx"]), "cy": float(r["cy"]),
                "width": int(r["width"]), "height": int(r["height"]),
            })
    return rows


def load_train_check_poses(sparse_dir: Path, n: int = 5):
    from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary, read_images_binary
    images = read_images_binary(sparse_dir / "images.bin")
    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    rows = []
    for i, (_, im) in enumerate(sorted(images.items(), key=lambda kv: kv[1].name)):
        if i % max(1, len(images) // n) != 0:
            continue
        cam = cameras[im.camera_id]
        # cam.params for PINHOLE: fx, fy, cx, cy (SIMPLE_PINHOLE: f, cx, cy)
        if len(cam.params) == 4:
            fx, fy, cx, cy = cam.params
        else:
            f, cx, cy = cam.params[:3]
            fx = fy = f
        rows.append({
            "image_name": im.name,
            "qvec": list(im.qvec), "tvec": list(im.tvec),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": cam.width, "height": cam.height,
        })
        if len(rows) >= n:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path, help="path to config.yml from a Nerfstudio run")
    ap.add_argument("--mode", choices=["test", "train_check"], required=True)
    ap.add_argument("--poses-csv", type=Path, help="test_poses.csv (mode=test)")
    ap.add_argument("--sparse-dir", type=Path, help="scene train/sparse/0 dir (mode=train_check)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-train-check", type=int, default=5)
    args = ap.parse_args()

    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.cameras.cameras import Cameras, CameraType

    config, pipeline, _, _ = eval_setup(args.config)
    dp_outputs = pipeline.datamanager.train_dataparser_outputs
    transform = dp_outputs.dataparser_transform.cpu().numpy()
    scale = float(dp_outputs.dataparser_scale)

    if args.mode == "test":
        assert args.poses_csv is not None
        rows = load_test_poses(args.poses_csv)
    else:
        assert args.sparse_dir is not None
        rows = load_train_check_poses(args.sparse_dir, args.n_train_check)

    args.out.mkdir(parents=True, exist_ok=True)
    device = pipeline.device

    for row in rows:
        c2w = colmap_pose_to_c2w(row["qvec"], row["tvec"])
        c2w = apply_dataparser_transform(c2w, transform, scale)
        c2w_t = torch.tensor(c2w, dtype=torch.float32)[:3, :4].unsqueeze(0)

        camera = Cameras(
            camera_to_worlds=c2w_t,
            fx=torch.tensor([row["fx"]], dtype=torch.float32),
            fy=torch.tensor([row["fy"]], dtype=torch.float32),
            cx=torch.tensor([row["cx"]], dtype=torch.float32),
            cy=torch.tensor([row["cy"]], dtype=torch.float32),
            width=torch.tensor([row["width"]], dtype=torch.long),
            height=torch.tensor([row["height"]], dtype=torch.long),
            camera_type=CameraType.PERSPECTIVE,
        ).to(device)

        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera)
        rgb = outputs["rgb"].clamp(0, 1).cpu().numpy()
        img = Image.fromarray((rgb * 255).astype(np.uint8))
        img.save(args.out / row["image_name"])
        print("rendered", row["image_name"])


if __name__ == "__main__":
    main()
