"""Analysis 08: v1.1 remap fix for the two k=-0.115 scenes (HNI0131, HNI0265).

Their negative k means the distorted GT sees rays BEYOND the pinhole render's
FOV, so the v1.0 remap (06) edge-replicates a ~16-38 px border band (~6-8% of
pixels). Fix: re-render every test view on an expanded canvas (same focal,
W/H + 2*MARGIN, principal point shifted by MARGIN) so the remap always has real
source pixels, then remap directly into the distorted 1320x989 geometry and
overwrite the exp030 staging + rebuild the submission zip.

The checkpoints were trained on Kaggle, so config paths are rewritten to the
local staging/checkpoint dirs via eval_setup's update_config_callback.

Run: conda run -n airace python Analysis/08_hni_expanded_canvas_fix.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "phase1" / "private_set1"
STAGE = REPO / "submissions/phase1/exp030_distortion_remap_results/renders"
OUT_ROOT = REPO / "submissions/phase1/exp030_distortion_remap_results"
MARGIN = 128  # k=-0.115 needs ~87px at corners (measured by the in-loop assert at 64)
sys.path.insert(0, str(REPO))

from src.render import colmap_pose_to_c2w, apply_dataparser_transform, load_test_poses  # noqa: E402
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary  # noqa: E402

SCENES = ["HNI0131", "HNI0265"]


def bilinear(img, us, vs):
    H, W = img.shape[:2]
    u0 = np.clip(np.floor(us).astype(int), 0, W - 2)
    v0 = np.clip(np.floor(vs).astype(int), 0, H - 2)
    du = np.clip(us - u0, 0, 1)[..., None]
    dv = np.clip(vs - v0, 0, 1)[..., None]
    return (img[v0, u0] * (1 - du) * (1 - dv) + img[v0, u0 + 1] * du * (1 - dv)
            + img[v0 + 1, u0] * (1 - du) * dv + img[v0 + 1, u0 + 1] * du * dv)


def load_pipeline(scene: str):
    from nerfstudio.utils.eval_utils import eval_setup
    run_dir = sorted((REPO / f"runs/phase1/exp005_antialiased_dense/{scene}/train_staging_dense/splatfacto").iterdir())[-1]
    local_data = REPO / f"data/processed/phase1/{scene}/train_staging_dense"

    def fix_paths(config):
        config.output_dir = REPO / f"runs/phase1/exp005_antialiased_dense/{scene}"
        config.pipeline.datamanager.data = local_data
        if hasattr(config.pipeline.datamanager, "dataparser"):
            config.pipeline.datamanager.dataparser.data = local_data
        return config

    config, pipeline, _, _ = eval_setup(run_dir / "config.yml", update_config_callback=fix_paths)
    dp = pipeline.datamanager.train_dataparser_outputs
    return pipeline, dp.dataparser_transform.cpu().numpy(), float(dp.dataparser_scale)


def main():
    for scene in SCENES:
        cams = read_cameras_binary(RAW / scene / "train/sparse/0/cameras.bin")
        cam = list(cams.values())[0]
        f, cx, cy, k = cam.params
        assert k < -0.05, f"{scene}: expected large negative k, got {k}"
        pipeline, transform, scale = load_pipeline(scene)
        device = pipeline.device
        rows = load_test_poses(RAW / scene / "test/test_poses.csv")
        dst = STAGE / scene / "renders_test"
        assert dst.exists()

        from nerfstudio.cameras.cameras import Cameras, CameraType
        for r in rows:
            W, H = r["width"], r["height"]
            We, He = W + 2 * MARGIN, H + 2 * MARGIN
            c2w = colmap_pose_to_c2w(r["qvec"], r["tvec"])
            c2w = apply_dataparser_transform(c2w, transform, scale)
            camera = Cameras(
                camera_to_worlds=torch.tensor(c2w, dtype=torch.float32)[:3, :4].unsqueeze(0),
                fx=torch.tensor([r["fx"]], dtype=torch.float32),
                fy=torch.tensor([r["fy"]], dtype=torch.float32),
                cx=torch.tensor([r["cx"] + MARGIN], dtype=torch.float32),
                cy=torch.tensor([r["cy"] + MARGIN], dtype=torch.float32),
                width=torch.tensor([We], dtype=torch.long),
                height=torch.tensor([He], dtype=torch.long),
                camera_type=CameraType.PERSPECTIVE,
            ).to(device)
            with torch.no_grad():
                out = pipeline.model.get_outputs_for_camera(camera)
            big = out["rgb"].clamp(0, 1).cpu().numpy()

            # distorted output grid (original intrinsics), sample the expanded render
            u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
            xd = (u - r["cx"]) / r["fx"]
            yd = (v - r["cy"]) / r["fy"]
            xu, yu = xd.copy(), yd.copy()
            for _ in range(6):
                r2 = xu * xu + yu * yu
                xu = xd / (1 + k * r2)
                yu = yd / (1 + k * r2)
            us = xu * r["fx"] + r["cx"] + MARGIN
            vs = yu * r["fy"] + r["cy"] + MARGIN
            assert us.min() >= 0 and us.max() < We - 1 and vs.min() >= 0 and vs.max() < He - 1, \
                f"MARGIN={MARGIN} insufficient: u[{us.min():.0f},{us.max():.0f}] v[{vs.min():.0f},{vs.max():.0f}]"
            img = (np.clip(bilinear(big, us, vs), 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img).save(dst / r["image_name"], quality=98)
        print(f"{scene}: {len(rows)} views re-rendered on {2*MARGIN}px-expanded canvas and remapped")
        del pipeline
        torch.cuda.empty_cache()

    # repackage private partial + rebuild merged zip
    subprocess.run([sys.executable, str(REPO / "src/package_submission.py"),
                    "--runs-dir", str(STAGE),
                    "--scenes", "HCM0249", "HCM0254", "HCM0276", "HCM1439",
                    "HNI0131", "HNI0265", "HNI0366", "HNI0437",
                    "--poses-root", str(RAW),
                    "--out", str(OUT_ROOT / "partial_private_set1.zip")], check=True)
    import zipfile
    final = OUT_ROOT / "submission_round1.zip"
    with zipfile.ZipFile(final, "w", zipfile.ZIP_STORED) as zout:
        for part in ("partial_public_set.zip", "partial_private_set1.zip"):
            with zipfile.ZipFile(OUT_ROOT / part) as zin:
                for info in zin.infolist():
                    zout.writestr(info, zin.read(info))
    print(f"Rebuilt {final} ({final.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
