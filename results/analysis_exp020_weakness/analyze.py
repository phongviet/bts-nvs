import os, json, csv
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import lpips
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
import cv2

REPO = Path("/home/phong/Viettel_AI_Race_2026/bts-nvs")
SCENES = {
    "hcm0034": REPO / "runs/phase1/exp004_backend_ablation/antialiased/renders_test",
    "hcm0031": REPO / "runs/phase1/exp004_backend_ablation/hcm0031_antialiased/renders_test",
    "HCM0181": REPO / "runs/phase1/exp004_backend_ablation/HCM0181_antialiased/renders_test",
    "HCM0193": REPO / "runs/phase1/exp004_backend_ablation/HCM0193_antialiased/renders_test",
    # HCM0204 added 2026-07-10 once its antialiased checkpoint existed (exp004_hcm0204_fill);
    # the original run had to exclude it for lack of a local checkpoint.
    "HCM0204": REPO / "runs/phase1/exp004_hcm0204_fill/HCM0204/antialiased/renders_test",
}
GT_ROOT = REPO / "data/raw/phase1/public_set"
OUT = REPO / "results/analysis_exp020_weakness"

PSNR_MAX = 50.0
device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = lpips.LPIPS(net="vgg").to(device).eval()

def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))

def to_lpips_tensor(arr):
    t = torch.from_numpy(arr).float().permute(2,0,1)[None] / 127.5 - 1.0
    return t.to(device)

def laplacian_var(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

rows = []
tile_records = []  # per-image tile-error breakdown

for scene, render_dir in SCENES.items():
    gt_dir = GT_ROOT / scene / "test/images"
    files = sorted(p.name for p in render_dir.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))
    print(f"{scene}: {len(files)} images", flush=True)
    for fn in files:
        gt_p = gt_dir / fn
        rd_p = render_dir / fn
        if not gt_p.exists():
            continue
        gt = load_rgb(gt_p)
        rd = load_rgb(rd_p)
        if rd.shape != gt.shape:
            rd = np.asarray(Image.fromarray(rd).resize((gt.shape[1], gt.shape[0]), Image.LANCZOS))

        psnr = psnr_fn(gt, rd, data_range=255)
        ssim = ssim_fn(gt, rd, channel_axis=2, data_range=255)
        with torch.no_grad():
            lp = loss_fn(to_lpips_tensor(gt), to_lpips_tensor(rd)).item()
        psnr_norm = min(max(psnr / PSNR_MAX, 0.0), 1.0)
        score = 0.4*(1-lp) + 0.3*ssim + 0.3*psnr_norm

        gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        sharpness = laplacian_var(gt_gray)
        brightness = gt_gray.mean()

        rows.append(dict(scene=scene, file=fn, psnr=psnr, ssim=ssim, lpips=lp, score=score,
                          gt_sharpness=sharpness, gt_brightness=brightness,
                          width=gt.shape[1], height=gt.shape[0]))

        # tile-based error map: 6x4 grid, per-tile L2 error (proxy, cheap) + max-error tile location
        h, w = gt.shape[:2]
        ny, nx = 4, 6
        tile_errs = np.zeros((ny, nx))
        diff = (gt.astype(np.float32) - rd.astype(np.float32))
        for iy in range(ny):
            for ix in range(nx):
                y0, y1 = iy*h//ny, (iy+1)*h//ny
                x0, x1 = ix*w//nx, (ix+1)*w//nx
                tile_errs[iy, ix] = np.mean(diff[y0:y1, x0:x1]**2)
        tile_records.append(dict(scene=scene, file=fn, tile_errs=tile_errs.tolist(),
                                  max_tile=(int(np.unravel_index(tile_errs.argmax(), tile_errs.shape)[0]),
                                            int(np.unravel_index(tile_errs.argmax(), tile_errs.shape)[1]))))

with open(OUT/"per_image_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

json.dump(tile_records, open(OUT/"tile_errors.json", "w"))

print(f"\nTotal images scored: {len(rows)}")
print(f"Wrote {OUT/'per_image_metrics.csv'} and {OUT/'tile_errors.json'}")
