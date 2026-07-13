"""Analysis 13: single-encode check (E1b). The exp033 flow was: refiner output
-> save q98 JPEG -> re-encode q95 (two lossy generations). Here: re-apply the
refiner on the cached test inputs, keep float -> encode ONCE at q95 4:4:4
(+optimize) -> score. Also scores the lossless PNG as the pipeline ceiling.

Run: conda run -n airace python Analysis/13_single_encode_check.py --scene hcm0034 [--tta]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.metrics import compute_metrics  # noqa: E402

spec = importlib.util.spec_from_file_location("ref10", REPO / "Analysis/10_refiner_pilot.py")
ref10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref10)

OUT = REPO / "Analysis/X5_refiner"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="hcm0034")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--variant", default="")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scene = args.scene

    ckpt = OUT / scene / f"refiner{args.suffix}.pt"
    if not ckpt.exists():
        ckpt = OUT / scene / "refiner.pt"
    sd = torch.load(ckpt, map_location=device)
    net = ref10.UNet(base=sd["d1.net.0.weight"].shape[0]).to(device)
    net.load_state_dict(sd); net.eval()

    icache = OUT / scene / f"test_inputs{args.variant}"
    files = sorted(icache.glob("*.npz"))
    assert files, f"no cached test inputs at {icache}"
    d_png = OUT / scene / "enc_png"; d_png.mkdir(exist_ok=True)
    d_q95 = OUT / scene / "enc_q95sub0"; d_q95.mkdir(exist_ok=True)
    for fp in files:
        inp = np.load(fp)["inp"].astype(np.float32).transpose(2, 0, 1)
        img = ref10._net_apply(net, inp, device, tta=args.tta)
        pil = Image.fromarray((img * 255).astype(np.uint8))
        name = fp.name[:-4]  # strip .npz -> original image name (*.JPG)
        pil.save(d_png / (name + ".png"))
        pil.save(d_q95 / name, "JPEG", quality=95, subsampling=0, optimize=True)

    gt = ref10.dibr04.scene_raw(scene) / scene / "test/images"
    # rename PNGs to match GT names for the metric tool? compute_metrics matches
    # by filename; keep JPG-named q95 dir as the headline, PNG scored via map.
    m95 = compute_metrics(d_q95, gt, "vgg", 50.0)["mean"]
    print(f"{scene} single-encode q95_sub0 (tta={args.tta}): "
          f"Score={m95['score']:.5f} PSNR={m95['psnr']:.3f} "
          f"SSIM={m95['ssim']:.4f} LPIPS={m95['lpips']:.4f}")
    nbytes = sum(f.stat().st_size for f in d_q95.iterdir())
    print(f"  q95_sub0 bytes for scene: {nbytes/1e6:.1f} MB over {len(files)} imgs")


if __name__ == "__main__":
    main()
