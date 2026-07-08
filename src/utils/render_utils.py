"""Shared checkpoint-rendering helpers (used by run_experiment, render_val,
build_enhancer_pairs). Pose conventions per src/render.py's module docstring
-- the dataparser_transform already contains the world-coordinate swap; only
the OpenCV->OpenGL flip is applied manually.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.render import apply_dataparser_transform, colmap_pose_to_c2w

# COLMAP models whose params start (fx, fy, cx, cy); the SIMPLE_*/RADIAL
# family starts (f, cx, cy, ...) with a shared focal instead.
_TWO_FOCAL_MODELS = {"PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE",
                     "THIN_PRISM_FISHEYE", "FOV"}


def load_colmap_poses(sparse_dir: Path, only_names: set[str] | None = None) -> list[dict]:
    """Full camera rows (pose + intrinsics) from a COLMAP sparse dir."""
    from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary, read_images_binary
    images = read_images_binary(sparse_dir / "images.bin")
    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    rows = []
    for _, im in sorted(images.items(), key=lambda kv: kv[1].name):
        if only_names is not None and im.name not in only_names:
            continue
        cam = cameras[im.camera_id]
        if cam.model in _TWO_FOCAL_MODELS:
            fx, fy, cx, cy = cam.params[:4]
        else:
            fx, cx, cy = cam.params[:3]
            fy = fx
        rows.append({"image_name": im.name, "qvec": list(im.qvec), "tvec": list(im.tvec),
                     "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                     "width": cam.width, "height": cam.height})
    return rows


def render_pose_rows(config_path: Path, rows: list[dict], out_dir: Path,
                     quality: int = 98, skip_existing: bool = True) -> int:
    """Render camera rows from a trained checkpoint into out_dir (keeps names)."""
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from nerfstudio.utils.eval_utils import eval_setup

    _, pipeline, _, _ = eval_setup(config_path)
    dp = pipeline.datamanager.train_dataparser_outputs
    transform = dp.dataparser_transform.cpu().numpy()
    scale = float(dp.dataparser_scale)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pipeline.device

    n = 0
    for row in rows:
        out_path = out_dir / row["image_name"]
        if skip_existing and out_path.exists():
            continue
        c2w = apply_dataparser_transform(colmap_pose_to_c2w(row["qvec"], row["tvec"]),
                                         transform, scale)
        camera = Cameras(
            camera_to_worlds=torch.tensor(c2w, dtype=torch.float32)[:3, :4].unsqueeze(0),
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
        save_kwargs = {"quality": quality} if out_path.suffix.lower() in (".jpg", ".jpeg") else {}
        Image.fromarray((rgb * 255).astype(np.uint8)).save(out_path, **save_kwargs)
        n += 1
    return n
