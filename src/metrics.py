"""Compute per-image PSNR/SSIM/LPIPS and the competition combined Score.

Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm

PSNR_MAX = 50.0, CONFIRMED 2026-07-09: solved exactly (50.0001 / 50.0000)
from the leaderboard's per-metric breakdowns of submissions #3 and #4 --
see results/PROGRESS.md calibration section. Per-dB score weight is thus
0.3/50 = 0.006, not the 0.0075 assumed under the old 40.0 placeholder.
LPIPS backbone: leaderboard values (~0.27) run ~2x local alex (~0.13),
consistent with VGG -- see PROGRESS; prefer --lpips-net vgg for decisions.
"""
import argparse
import json
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img).astype(np.float32) / 255.0


def compute_metrics(render_dir: Path, gt_dir: Path, lpips_net: str, psnr_max: float) -> dict:
    loss_fn = lpips.LPIPS(net=lpips_net)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = loss_fn.to(device)

    rows = []
    render_files = sorted(render_dir.glob("*"))
    for rf in render_files:
        gf = gt_dir / rf.name
        if not gf.exists():
            continue
        gt = load_rgb(gf)
        pred = load_rgb(rf, size=(gt.shape[1], gt.shape[0]))

        psnr = float(sk_psnr(gt, pred, data_range=1.0))
        ssim = float(sk_ssim(gt, pred, data_range=1.0, channel_axis=2, win_size=11))

        t_gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        t_pred = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        with torch.no_grad():
            lp = float(loss_fn(t_gt, t_pred).item())

        psnr_norm = float(np.clip(psnr / psnr_max, 0.0, 1.0))
        score = 0.4 * (1 - lp) + 0.3 * ssim + 0.3 * psnr_norm

        rows.append({
            "image": rf.name, "psnr": psnr, "ssim": ssim, "lpips": lp,
            "psnr_norm": psnr_norm, "score": score,
        })

    if not rows:
        raise RuntimeError(f"No matching render/GT pairs found: {render_dir} vs {gt_dir}")

    mean = {
        "image": "mean",
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "lpips": float(np.mean([r["lpips"] for r in rows])),
        "psnr_norm": float(np.mean([r["psnr_norm"] for r in rows])),
        "score": float(np.mean([r["score"] for r in rows])),
    }
    rows.append(mean)
    return {"per_image": rows, "mean": mean}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True, type=Path, help="dir of rendered test images")
    ap.add_argument("--gt", required=True, type=Path, help="dir of ground-truth test images")
    ap.add_argument("--out", required=True, type=Path, help="output metrics json path")
    ap.add_argument("--lpips-net", default="alex", choices=["alex", "vgg"])
    ap.add_argument("--psnr-max", default=50.0, type=float,
                     help="confirmed 2026-07-09 from leaderboard metric breakdowns")
    args = ap.parse_args()

    result = compute_metrics(args.renders, args.gt, args.lpips_net, args.psnr_max)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    m = result["mean"]
    print(f"PSNR={m['psnr']:.3f} SSIM={m['ssim']:.4f} LPIPS={m['lpips']:.4f} "
          f"PSNR_norm={m['psnr_norm']:.4f} Score={m['score']:.4f}")


if __name__ == "__main__":
    main()
