"""Render a trained SSS model at competition test poses (test_poses.csv).

CSV poses are COLMAP world-to-camera (confirmed in bts-nvs
src/utils/pose_utils.py); principal points are exactly centered, so the
3DGS pinhole camera applies directly: R = qvec2rotmat(q).T, T = t.

Usage:
  python render_test_csv.py --model <model_dir> --poses-csv <test_poses.csv> \
      --out <renders_dir> [--iteration N]
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision

SSS = Path(__file__).parent / "3D-student-splatting-and-scooping"
sys.path.insert(0, str(SSS))

from scene.cameras import Camera  # noqa: E402
from scene.colmap_loader import qvec2rotmat  # noqa: E402
from scene.nt_model import NTModel  # noqa: E402
from t_renderer import render  # noqa: E402


class Pipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--poses-csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--iteration", type=int, default=None)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--nu-degree", type=float, default=100)
    args = ap.parse_args()

    pc_root = args.model / "point_cloud"
    if args.iteration is None:
        args.iteration = max(int(d.name.split("_")[1]) for d in pc_root.iterdir())
    ply = pc_root / f"iteration_{args.iteration}" / "point_cloud.ply"

    model = NTModel(args.sh_degree, args.nu_degree)
    model.load_ply(str(ply))
    print(f"loaded {ply} ({model.get_xyz.shape[0]} components, "
          f"active_sh_degree={model.active_sh_degree})")

    args.out.mkdir(parents=True, exist_ok=True)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    with open(args.poses_csv) as f:
        rows = list(csv.DictReader(f))
    with torch.no_grad():
        for i, r in enumerate(rows):
            q = [float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])]
            t = np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])])
            W, H = int(r["width"]), int(r["height"])
            fovx = 2 * math.atan(W / (2 * float(r["fx"])))
            fovy = 2 * math.atan(H / (2 * float(r["fy"])))
            cam = Camera(colmap_id=i, R=np.transpose(qvec2rotmat(q)), T=t,
                         FoVx=fovx, FoVy=fovy,
                         image=torch.zeros(3, H, W), gt_alpha_mask=None,
                         image_name=r["image_name"], uid=i, data_device="cpu")
            img = torch.clamp(render(cam, model, Pipe(), background)["render"], 0.0, 1.0)
            # Save via PIL at quality=98 to MATCH the splatfacto control
            # (src/utils/render_utils.render_pose_rows default q98). torchvision's
            # save_image defaults to PIL q75, which would unfairly cost SSS ~0.003
            # LPIPS and invalidate the gate-1 comparison.
            arr = (img.mul(255).clamp(0, 255).byte().permute(1, 2, 0)
                   .cpu().numpy())
            from PIL import Image as _PILImage
            _out = args.out / r["image_name"]
            _sk = {"quality": 98} if _out.suffix.lower() in (".jpg", ".jpeg") else {}
            _PILImage.fromarray(arr).save(_out, **_sk)
            print(f"{i+1}/{len(rows)} {r['image_name']}")
    print("done:", args.out)


if __name__ == "__main__":
    main()
