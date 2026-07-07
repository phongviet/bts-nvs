"""Build the enhancer training-pair dataset (Week-2b Day 4 -> Week-3 input).

For each scene: render the val-split poses (from splits/val_ids.txt) with the
scene's current-best trained checkpoint, pair each render with its real train
image, compute per-pair LPIPS, and register everything in one manifest.

Output layout:
  data/processed/phase1/enhancer_pairs/
    render/<scene>__<image_name>     degraded input (our render)
    real/<scene>__<image_name>       clean target (real photo)
    mask/<scene>__<image_name>.png   optional transient mask (if built)
    manifest.csv                     scene,image_name,render,real,mask,lpips

Pairs come ONLY from the organizers' provided images rendered/photographed at
the same poses -- compliant with the no-external-data rule (the manifest is
the provenance record for enhancer finetuning).

Usage (per scene; loop scenes in the caller):
  python src/enhancer/build_enhancer_pairs.py \
      --config runs/phase1/exp005_antialiased_dense/HCM0249/.../config.yml \
      --scene-dir data/raw/phase1/private_set1/HCM0249 \
      --processed-root data/processed/phase1 \
      --out-root data/processed/phase1/enhancer_pairs
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.render import apply_dataparser_transform, colmap_pose_to_c2w  # noqa: E402

MANIFEST_FIELDS = ["scene", "image_name", "render", "real", "mask", "lpips"]


def load_val_poses(sparse_dir: Path, val_ids: set[str]):
    from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary, read_images_binary
    images = read_images_binary(sparse_dir / "images.bin")
    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    rows = []
    for _, im in sorted(images.items(), key=lambda kv: kv[1].name):
        if im.name not in val_ids:
            continue
        cam = cameras[im.camera_id]
        if len(cam.params) == 4:
            fx, fy, cx, cy = cam.params
        else:
            fx, cx, cy = cam.params[:3]
            fy = fx
        rows.append({"image_name": im.name, "qvec": list(im.qvec), "tvec": list(im.tvec),
                     "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                     "width": cam.width, "height": cam.height})
    return rows


def render_rows(config_path: Path, rows: list[dict], out_dir: Path):
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from nerfstudio.utils.eval_utils import eval_setup

    config, pipeline, _, _ = eval_setup(config_path)
    dp = pipeline.datamanager.train_dataparser_outputs
    transform = dp.dataparser_transform.cpu().numpy()
    scale = float(dp.dataparser_scale)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pipeline.device

    for row in rows:
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
        Image.fromarray((rgb * 255).astype(np.uint8)).save(out_dir / row["image_name"], quality=98)
        print("rendered", row["image_name"])


def lpips_per_pair(render_dir: Path, real_dir: Path, names: list[str], net: str = "alex"):
    import lpips as lpips_mod
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips_mod.LPIPS(net=net).to(device)
    out = {}
    for n in names:
        a = np.asarray(Image.open(render_dir / n).convert("RGB"), dtype=np.float32) / 255
        b = np.asarray(Image.open(real_dir / n).convert("RGB"), dtype=np.float32) / 255
        ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        with torch.no_grad():
            out[n] = float(loss_fn(ta, tb).item())
    return out


def upsert_manifest(manifest: Path, scene: str, new_rows: list[dict]):
    rows = []
    if manifest.exists():
        with open(manifest) as f:
            rows = [r for r in csv.DictReader(f) if r["scene"] != scene]
    rows += new_rows
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path, help="trained run's config.yml")
    ap.add_argument("--scene-dir", required=True, type=Path)
    ap.add_argument("--processed-root", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--lpips-net", default="alex")
    args = ap.parse_args()

    scene = args.scene_dir.name
    scene_root = args.processed_root / scene
    val_ids_file = scene_root / "splits" / "val_ids.txt"
    if not val_ids_file.exists():
        raise SystemExit(f"{val_ids_file} missing -- run make_val_split.py (match-test) first.")
    val_ids = set(val_ids_file.read_text().split())

    rows = load_val_poses(scene_root / "colmap_train_only", val_ids)
    if not rows:
        raise SystemExit(f"{scene}: no val poses found in colmap_train_only for {len(val_ids)} ids")

    tmp_render = args.out_root / "_tmp_render" / scene
    render_rows(args.config, rows, tmp_render)

    render_dir = args.out_root / "render"
    real_dir = args.out_root / "real"
    mask_out = args.out_root / "mask"
    for d in (render_dir, real_dir, mask_out):
        d.mkdir(parents=True, exist_ok=True)

    names = [r["image_name"] for r in rows]
    lp = lpips_per_pair(tmp_render, args.scene_dir / "train" / "images", names, args.lpips_net)

    manifest_rows = []
    masks_dir = scene_root / "transient_masks"
    for n in names:
        key = f"{scene}__{n}"
        shutil.copy2(tmp_render / n, render_dir / key)
        shutil.copy2(args.scene_dir / "train" / "images" / n, real_dir / key)
        mask_rel = ""
        msk = masks_dir / (Path(n).stem + ".png")
        if msk.exists():
            shutil.copy2(msk, mask_out / (key + ".png"))
            mask_rel = f"mask/{key}.png"
        manifest_rows.append({"scene": scene, "image_name": n,
                              "render": f"render/{key}", "real": f"real/{key}",
                              "mask": mask_rel, "lpips": round(lp[n], 5)})
    shutil.rmtree(tmp_render)

    upsert_manifest(args.out_root / "manifest.csv", scene, manifest_rows)
    vals = [r["lpips"] for r in manifest_rows]
    print(f"{scene}: {len(manifest_rows)} pairs, LPIPS mean={np.mean(vals):.4f} "
          f"p90={np.percentile(vals, 90):.4f} -> {args.out_root}")


if __name__ == "__main__":
    main()
