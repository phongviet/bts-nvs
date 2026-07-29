"""Analysis 16 (exp037): budget-optimal JPEG encoding with per-image knapsack
rate allocation + optional mozjpeg trellis quantization.

Replaces the flat "one quality for the whole private set" step-down in
14_build_v2_submission.py. The competition Score is the per-image MEAN of
0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR/50, and the byte budget is a HARD total
cap (<=350 MB). That is a classic knapsack: given each image's quality ->
(size, quality-of-encode) curve, spend the fixed byte budget to maximise the
SUM of per-image scores.

Objective is GT-FREE (works on private scenes): JPEG can only DEGRADE the
lossless render, so we allocate bytes to minimise encode-induced Score loss,
measured as the self-reference fidelity of the decoded JPEG vs the lossless
render (SSIM + PSNR terms; LPIPS-tuned tables were shown NOT to help --
arXiv 2510.10970 -- so the proxy is SSIM/PSNR, matching mozjpeg's -tune-ms-ssim).

Greedy Lagrangian: start every image at the lowest quality that fits, then
repeatedly bump the image whose next quality step buys the most
Score-per-byte, until the next bump would break the budget. This equalises
marginal value across images (busy/low-SSIM frames get more bytes than sky).

Encoder backends, best first:
  * mozjpeg `cjpeg` on PATH  -> trellis + progressive + -tune-ms-ssim
  * Pillow (libjpeg-turbo)   -> optimize=True, subsampling=0 (4:4:4)

Run (standalone A/B on a render dir):
  conda run -n airace python Analysis/16_encode_knapsack.py \
      --renders Analysis/X5_refiner/hcm0034/renders_refined_v2 \
      --budget-mb 60 --out /tmp/enc_hcm0034
"""
from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

def _find_mozjpeg():
    """mozjpeg ships its encoder as `cjpeg`, but so does libjpeg-turbo (whose
    cjpeg has NO trellis quantization and lacks -tune-*). Only accept the binary
    if `-version` self-identifies as mozjpeg -- otherwise Pillow (also
    libjpeg-turbo) is identical, so we skip the subprocess overhead."""
    cand = shutil.which("cjpeg")
    if cand is None:
        return None
    try:
        v = subprocess.run([cand, "-version"], capture_output=True, text=True)
        blob = (v.stdout + v.stderr).lower()
        return cand if "mozjpeg" in blob else None
    except Exception:
        return None


MOZJPEG = _find_mozjpeg()


# ------------------------------- fidelity proxy --------------------------------
def ssim_np(a, b):
    """Mean SSIM over channels; a,b float [0,1] HxWx3. Uses cv2 Gaussian
    filtering (fast; the proxy only needs to RANK quality steps, so an
    11x11 sigma=1.5 window on the full image is ample)."""
    import cv2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ks, sig = (11, 11), 1.5
    blur = lambda x: cv2.GaussianBlur(x, ks, sig)
    mu_a, mu_b = blur(a), blur(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = blur(a * a) - mu_a2
    sb = blur(b * b) - mu_b2
    sab = blur(a * b) - mu_ab
    s = ((2 * mu_ab + c1) * (2 * sab + c2)) / ((mu_a2 + mu_b2 + c1) * (sa + sb + c2))
    return float(s.mean())


def psnr_np(a, b):
    mse = float(((a - b) ** 2).mean())
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def _downsample(im: Image.Image, maxside: int) -> np.ndarray:
    w, h = im.size
    s = maxside / max(w, h)
    if s < 1.0:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


def recoverable_score(decoded, ref, maxside=640):
    """Self-reference Score fraction retained by the encode (higher = better).
    Uses only the SSIM + PSNR terms of the grader (0.3 each); the LPIPS term is
    omitted on purpose (expensive, and LPIPS-tuned encoding does not help).
    Computed on a downsampled copy (<= maxside) -- the quality RANKING across
    encode settings is preserved and it is ~6x cheaper than full-res."""
    a = _downsample(decoded, maxside)
    b = _downsample(ref, maxside)
    return 0.3 * ssim_np(a, b) + 0.3 * min(psnr_np(a, b), 50.0) / 50.0


# --------------------------------- encoders ------------------------------------
def encode(im: Image.Image, quality: int, subsampling=0, tune="ms-ssim") -> bytes:
    """Encode a PIL image to JPEG bytes. mozjpeg (trellis) if available, else
    Pillow/libjpeg-turbo. subsampling 0 = 4:4:4."""
    if MOZJPEG is not None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.ppm"
            im.convert("RGB").save(src)  # PPM: lossless hand-off to cjpeg
            cmd = [MOZJPEG, "-quality", str(quality), "-optimize", "-progressive"]
            if subsampling == 0:
                cmd += ["-sample", "1x1"]
            if tune:
                cmd += ["-tune-" + tune]
            cmd.append(str(src))
            out = subprocess.run(cmd, capture_output=True, check=True)
            return out.stdout
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality, subsampling=subsampling,
                           optimize=True)
    return buf.getvalue()


def _decode(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


# ------------------------------- the knapsack ----------------------------------
def allocate(images, budget_bytes, qualities=range(88, 99, 2), subsampling=0,
             tune="ms-ssim", refs=None, verbose=True, weights=None):
    """images: [(name, PIL)]. refs: optional [(name, PIL)] lossless references
    for the fidelity proxy (defaults to `images` themselves = self-reference).
    Returns {name: (jpeg_bytes, quality)} maximising summed recoverable_score
    under the total byte budget.

    weights: optional {name: float} on each frame's marginal score gain. Default
    (all 1.0) maximises the mean over FRAMES. The round-2 grader averages over
    SCENES, so the correct weight is 1/(n_scenes * frames_in_that_scene) -- e.g.
    a 28-frame bonsai frame is worth 2.14x a 60-frame drone frame. Left
    unweighted, the pooled knapsack quietly spends the budget on whichever scene
    has the most frames.

    Curve per image: encode at each quality once, record (size, score). Greedy
    upgrade from the cheapest rung by best Score-per-extra-byte until no upgrade
    fits the remaining budget."""
    qs = sorted(set(int(q) for q in qualities))
    weights = weights or {}
    ref_map = {n: im for n, im in (refs or images)}
    curves = {}  # name -> list of (quality, bytes, score) sorted by size asc
    for name, im in images:
        ref = ref_map.get(name, im)
        pts = []
        for q in qs:
            data = encode(im, q, subsampling, tune)
            pts.append((q, data, len(data), recoverable_score(_decode(data), ref)))
        # keep only Pareto-improving rungs (size up must buy score up)
        pts.sort(key=lambda t: t[2])
        pruned, best_s = [], -1.0
        for q, data, sz, sc in pts:
            if sc > best_s:
                pruned.append((q, data, sz, sc))
                best_s = sc
        curves[name] = pruned

    # start everyone at their smallest rung
    chosen = {n: 0 for n in curves}
    total = sum(curves[n][0][2] for n in curves)
    if total > budget_bytes:
        if verbose:
            print(f"  WARN: floor {total/1e6:.1f} MB already over "
                  f"{budget_bytes/1e6:.1f} MB budget -> shipping min-quality")
    else:
        # greedy marginal-gain upgrades
        while True:
            best = None  # (ratio, name)
            for n, ci in chosen.items():
                rungs = curves[n]
                if ci + 1 >= len(rungs):
                    continue
                dsz = rungs[ci + 1][2] - rungs[ci][2]
                dsc = rungs[ci + 1][3] - rungs[ci][3]
                if dsz <= 0 or total + dsz > budget_bytes:
                    continue
                # Weighted marginal gain. Unweighted (all w=1) maximises the mean
                # over FRAMES, but the grader averages over SCENES -- so a scene
                # with few frames is worth more per frame, and equal weighting
                # silently starves it. See `weights` in the docstring.
                ratio = dsc * weights.get(n, 1.0) / dsz
                if best is None or ratio > best[0]:
                    best = (ratio, n, dsz)
            if best is None:
                break
            _, n, dsz = best
            chosen[n] += 1
            total += dsz

    out = {}
    for n, ci in chosen.items():
        q, data, sz, sc = curves[n][ci]
        out[n] = (data, q)
    if verbose:
        used = sum(len(d) for d, _ in out.values())
        qhist = {}
        for _, q in out.values():
            qhist[q] = qhist.get(q, 0) + 1
        print(f"  knapsack: {used/1e6:.1f}/{budget_bytes/1e6:.1f} MB, "
              f"quality histogram {dict(sorted(qhist.items()))}, "
              f"backend={'mozjpeg' if MOZJPEG else 'pillow'}")
    return out


def load_dir(d: Path):
    imgs = []
    for f in sorted(list(d.glob("*.png")) + list(d.glob("*.JPG")) + list(d.glob("*.jpg"))):
        name = (f.stem + ".JPG") if f.suffix.lower() == ".png" else f.name
        imgs.append((name, Image.open(f).convert("RGB")))
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True, help="dir of PNG/JPG renders")
    ap.add_argument("--budget-mb", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qmin", type=int, default=88)
    ap.add_argument("--qmax", type=int, default=98)
    ap.add_argument("--qstep", type=int, default=2)
    ap.add_argument("--subsampling", type=int, default=0)
    ap.add_argument("--tune", default="ms-ssim")
    args = ap.parse_args()
    imgs = load_dir(Path(args.renders))
    assert imgs, f"no images under {args.renders}"
    alloc = allocate(imgs, int(args.budget_mb * 1e6),
                     qualities=range(args.qmin, args.qmax + 1, args.qstep),
                     subsampling=args.subsampling, tune=args.tune)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (data, q) in alloc.items():
        (out / name).write_bytes(data)
    print(f"wrote {len(alloc)} JPEGs to {out}")


if __name__ == "__main__":
    main()
