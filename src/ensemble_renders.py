"""exp012 (A.8): ensemble multiple render dirs of the SAME poses by per-pixel
averaging (mean, or median with >=3 inputs). Inputs are renders from the last
2-3 checkpoints of one run and/or different seeds. Averaging trades a little
sharpness (LPIPS risk) for PSNR/SSIM -- gate per scene like everything else.

Averaging happens in float over the decoded pixels; output encoding is a
postprocess encoder name (default jpeg95 -- pair the decision with exp011's
winner).

Usage:
  python src/ensemble_renders.py --inputs runA/renders_test runB/renders_test \
      --out runs/.../renders_test_ens [--mode mean|median] [--encoder jpeg95]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.postprocess.ops import ENCODERS  # noqa: E402


def ensemble(input_dirs: list[Path], out_dir: Path, mode: str = "mean",
             encoder: str = "jpeg95") -> int:
    names = None
    for d in input_dirs:
        cur = sorted(p.name for p in d.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if names is None:
            names = cur
        elif cur != names:
            raise SystemExit(f"Input dirs disagree on filenames: {d} has {len(cur)} vs {len(names)}; "
                             f"first diff: {sorted(set(cur) ^ set(names))[:3]}")
    if not names:
        raise SystemExit("No images found.")
    if mode == "median" and len(input_dirs) < 3:
        raise SystemExit("median needs >=3 inputs (with 2 it degenerates to mean).")

    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(ENCODERS[encoder])
    for name in names:
        stack = np.stack([np.asarray(Image.open(d / name).convert("RGB"), dtype=np.float32)
                          for d in input_dirs])
        avg = np.median(stack, axis=0) if mode == "median" else stack.mean(axis=0)
        Image.fromarray(avg.round().clip(0, 255).astype(np.uint8)).save(out_dir / name, **kwargs)
    return len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, nargs="+", type=Path,
                    help=">=2 render dirs with identical filenames")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", choices=["mean", "median"], default="mean")
    ap.add_argument("--encoder", default="jpeg95", choices=sorted(ENCODERS.keys()))
    args = ap.parse_args()
    if len(args.inputs) < 2:
        ap.error("need >=2 input dirs")
    n = ensemble(args.inputs, args.out, args.mode, args.encoder)
    print(f"Ensembled {len(args.inputs)} dirs ({args.mode}) -> {args.out}: {n} images")


if __name__ == "__main__":
    main()
