"""exp015/exp016 inference: run Difix3D+ (nvidia/difix, HF) over a render dir.

Difix is a single-step SD-Turbo-based image-to-image "fixer" (CVPR 2025,
github.com/nv-tlabs/Difix3D). Loaded via diffusers with trust_remote_code --
FIRST RUN on a rented GPU must verify (and record in the experiment log):
  1. the pipeline signature matches the model card
     (prompt="remove degradation", num_inference_steps=1, timesteps=[199],
      guidance_scale=0.0);
  2. resolution round-trip: competition frames are 1320x989; 989 % 8 != 0, so
     inputs are reflect-padded to /8 and cropped back EXACTLY (checked here);
  3. VRAM/latency at full res (tile via --tile if it OOMs).

Provenance: generic pretrained model, not trained on BTS/telecom scenes --
add the provenance row in docs/rules_and_constraints.md the day this enters
the repo (P3 standing duty). LoRA weights from train_difix_lora.py attach via
--lora.

Usage:
  python src/enhancer/run_difix.py --src <renders_dir> --dst <out_dir> \
      [--model nvidia/difix] [--lora runs/difix_lora/checkpoint-XXX] \
      [--prompt "remove degradation"] [--tile 0]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def pad_to_multiple(img: Image.Image, mult: int = 8) -> tuple[Image.Image, tuple[int, int]]:
    w, h = img.size
    pw = (mult - w % mult) % mult
    ph = (mult - h % mult) % mult
    if pw == 0 and ph == 0:
        return img, (w, h)
    arr = np.asarray(img)
    arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return Image.fromarray(arr), (w, h)


def load_pipeline(model_id: str, lora: Path | None, device: str):
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(model_id, trust_remote_code=True,
                                             torch_dtype=torch.float16)
    if lora is not None:
        pipe.load_lora_weights(str(lora))
        print(f"loaded LoRA weights from {lora}")
    pipe.to(device)
    return pipe


def enhance_image(pipe, img: Image.Image, prompt: str) -> Image.Image:
    padded, (w, h) = pad_to_multiple(img)
    out = pipe(prompt, image=padded, num_inference_steps=1, timesteps=[199],
               guidance_scale=0.0).images[0]
    if out.size != padded.size:
        # model returned a different resolution -- resize back before cropping,
        # and complain loudly: pixel alignment vs GT must be re-verified.
        print(f"WARNING: pipeline returned {out.size}, expected {padded.size}; resizing back. "
              f"Verify pixel alignment against GT before trusting metrics.")
        out = out.resize(padded.size, Image.LANCZOS)
    return out.crop((0, 0, w, h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--model", default="nvidia/difix")
    ap.add_argument("--lora", type=Path, default=None)
    ap.add_argument("--prompt", default="remove degradation")
    ap.add_argument("--quality", type=int, default=98, help="output JPEG quality")
    ap.add_argument("--limit", type=int, default=0, help="only first N images (smoke test)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(args.model, args.lora, device)

    args.dst.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in args.src.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.limit:
        paths = paths[:args.limit]
    import time
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        t0 = time.time()
        out = enhance_image(pipe, img, args.prompt)
        assert out.size == img.size, f"round-trip size mismatch on {p.name}: {out.size} vs {img.size}"
        save_kwargs = {"quality": args.quality} if p.suffix.lower() in (".jpg", ".jpeg") else {}
        out.save(args.dst / p.name, **save_kwargs)
        if i == 0:
            vram = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0
            print(f"first image: {time.time()-t0:.2f}s, peak VRAM {vram:.2f} GiB")
        print("enhanced", p.name)
    print(f"{len(paths)} images -> {args.dst}")


if __name__ == "__main__":
    main()
