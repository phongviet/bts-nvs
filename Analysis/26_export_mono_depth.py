"""Export Depth-Anything-V2 relative-depth maps for a staging dir's train images.

Backbone-side plan §2.2 (Analysis/PLAN_backbone_side_2026-07-26.md). The SoccerNet 2026
NVS winner (DENSER) used scale-and-shift-invariant DAv2 supervision to regularise geometry
in textureless regions; chair's recorded failure is exactly that -- MVS ran out of confident
geometry (smallest fused.ply of all 7 scenes).

We store the RAW relative inverse-depth (float16 .npy, one per image, same stem). No scale
alignment happens here on purpose: the loss is scale-and-shift invariant, so aligning now
would only bake in mono-depth's ambiguity. Consumers align per-step.

Usage:
  conda run -n airace python Analysis/26_export_mono_depth.py \
      --staging data/processed/round2/chair/train_staging_holdout
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", required=True, type=Path)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out-name", default="mono_depth")
    args = ap.parse_args()

    src = args.staging / "images"
    dst = args.staging / args.out_name
    dst.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not imgs:
        raise SystemExit(f"no images under {src}")

    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("depth-estimation", model=args.model, device=dev)

    done = 0
    for p in imgs:
        out = dst / f"{p.stem}.npy"
        if out.exists():
            done += 1
            continue
        im = Image.open(p).convert("RGB")
        d = pipe(im)["predicted_depth"]          # relative INVERSE depth, unnormalised
        # transformers returns (h,w) here, older/other versions (1,h,w); interpolate needs 4D.
        d = d.float().reshape(1, 1, *d.shape[-2:])
        d = torch.nn.functional.interpolate(
            d, size=(im.height, im.width), mode="bicubic", align_corners=False
        )[0, 0]
        np.save(out, d.cpu().numpy().astype(np.float16))
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(imgs)}", flush=True)

    # A constant map would make the loss a no-op with no error -- the exp009 failure shape.
    sample = np.load(dst / f"{imgs[0].stem}.npy").astype(np.float32)
    assert sample.shape == (Image.open(imgs[0]).height, Image.open(imgs[0]).width), \
        f"depth {sample.shape} != image size"
    assert float(sample.std()) > 1e-3, f"depth map is ~constant (std {sample.std():.2e}) -- useless"
    print(f"wrote {done} depth maps -> {dst}  (sample {sample.shape}, "
          f"range {sample.min():.2f}..{sample.max():.2f}, std {sample.std():.3f})")
    npy_to_png16(dst)



def npy_to_png16(depth_dir: Path):
    """Convert the raw .npy maps to the 16-bit PNGs nerfstudio's ColmapDataParser expects
    (`--depths-path <dir>`, filename forced to <image_stem>.png).

    Per-image min-max normalisation is INFORMATION-PRESERVING here because the consumer loss
    is scale-and-shift invariant: any affine transform of a relative depth map is absorbed by
    the per-step least-squares alignment. This buys nerfstudio's index-aligned `depth_filenames`
    ordering instead of us guessing the dataparser's image order.
    """
    n = 0
    for f in sorted(depth_dir.glob("*.npy")):
        d = np.load(f).astype(np.float32)
        lo, hi = float(d.min()), float(d.max())
        assert hi > lo, f"{f.name}: constant depth map"
        u16 = ((d - lo) / (hi - lo) * 65535.0).round().astype(np.uint16)
        Image.fromarray(u16, mode="I;16").save(depth_dir / f"{f.stem}.png")
        n += 1
    print(f"converted {n} maps -> 16-bit PNG in {depth_dir}")

if __name__ == "__main__":
    main()
