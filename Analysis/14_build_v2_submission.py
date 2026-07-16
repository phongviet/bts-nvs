"""Analysis 14: build the exp034 submission — refiner v2 + TTA, SINGLE-encoded
at the budget-optimal JPEG config (q95 4:4:4 +optimize; measured E1: +0.001 vs
shipped q95 4:2:0 and another +0.001 from single- vs double-encoding).

Per scene: load refiner{--suffix}.pt + the cached test inputs (variant-keyed),
re-apply the net (hflip TTA) in float, encode ONCE. Falls back to the v1
refiner/DIBR/remap render chain for scenes without a v2 checkpoint, so partial
fleets still build. Budget check: private zip must be <= 340 MB (350 rule with
slack); on overflow, quality steps down (95 -> 94 -> 93 ...) at 4:4:4 which E1
showed beats dropping chroma.

Run: conda run -n airace python Analysis/14_build_v2_submission.py
     [--suffix _v2] [--variant _ss2_cub] [--out exp034_v2_results]
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

def _load_by_path(mod: str, rel: str):
    spec = importlib.util.spec_from_file_location(mod, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ref10 = _load_by_path("ref10", "Analysis/10_refiner_pilot.py")
knap16 = _load_by_path("knap16", "Analysis/16_encode_knapsack.py")

X5 = REPO / "Analysis/X5_refiner"
X3 = REPO / "Analysis/X3_dibr"
EXP030 = REPO / "submissions/phase1/exp030_distortion_remap_results/renders"
RAW_PUB = REPO / "data/raw/phase1/public_set"
RAW_PRIV = REPO / "data/raw/phase1/private_set1"

PUBLIC = ["hcm0031", "hcm0034", "HCM0181", "HCM0193", "HCM0204"]
PRIVATE = ["HCM0249", "HCM0254", "HCM0276", "HCM1439", "HNI0131", "HNI0265", "HNI0366", "HNI0437"]
BUDGET_MB = 340


def png_renders(scene: str, suffix: str) -> list[tuple[str, Image.Image]]:
    """Lossless PNG renders (Kaggle fleet output or local --png apply). PNG
    stems are the original test names minus .JPG — restore that name."""
    d = X5 / scene / f"renders_refined{suffix}"
    pngs = sorted(d.glob("*.png")) if d.exists() else []
    if not pngs:
        return None
    return [(f.stem + ".JPG", Image.open(f).convert("RGB")) for f in pngs]


def apply_v2(scene: str, suffix: str, variant: str, device) -> list[tuple[str, Image.Image]]:
    """Returns [(image_name, PIL image)] from the v2 ckpt + cached inputs, or None."""
    ckpt = X5 / scene / f"refiner{suffix}.pt"
    icache = X5 / scene / f"test_inputs{variant}"
    files = sorted(icache.glob("*.npz"))
    if not (ckpt.exists() and files):
        return None
    net = ref10.load_refiner(ckpt, device)  # handles v3-wrapped and legacy raw sd
    out = []
    for fp in files:
        inp = np.load(fp)["inp"].astype(np.float32).transpose(2, 0, 1)
        img = ref10._net_apply(net, inp, device, tta=True)
        out.append((fp.name[:-4], Image.fromarray((img * 255).astype(np.uint8))))
    del net
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def fallback_images(scene: str) -> list[tuple[str, Image.Image]]:
    for d in (X5 / scene / "renders_refined", X3 / scene / "renders_g0.18",
              EXP030 / scene / "renders_test"):
        imgs = sorted(list(d.glob("*.JPG")) + list(d.glob("*.jpg"))) if d.exists() else []
        if imgs:
            print(f"  {scene}: FALLBACK source {d.relative_to(REPO)}")
            return [(f.name, Image.open(f).convert("RGB")) for f in imgs]
    raise FileNotFoundError(f"no render source for {scene}")


def encode_all(imgs, quality):
    enc = {}
    for name, im in imgs:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality, subsampling=0, optimize=True)
        enc[name] = buf.getvalue()
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v2")
    ap.add_argument("--variant", default="_ss2_cub")
    ap.add_argument("--out", default="exp034_v2_results")
    ap.add_argument("--encode", choices=["knapsack", "flat"], default="knapsack",
                    help="exp037 per-image knapsack rate allocation, or the "
                         "pre-exp037 flat q95->q90 step-down")
    # qmin defaults to the SHIPPED flat quality, not to the widest range.
    # MEASURED (Analysis/19_encode_ab.py, hcm0034, real grader): letting the
    # knapsack drop images BELOW q95 to fund q96 elsewhere scores -0.00017 at
    # equal bytes — the allocation proxy omits LPIPS (0.4 of Score) and LPIPS is
    # exactly what moved against us. Flooring at the shipped q makes this
    # non-negative BY CONSTRUCTION (no image lands below what exp034 shipped) and
    # leaves the part that actually pays: spending the ~18 MB of headroom a
    # whole-q ladder cannot reach (q95=321.7 MB, q96=353 MB > cap). That
    # measures +0.00042 on the same scene, with all three metrics improving.
    ap.add_argument("--qmin", type=int, default=95,
                    help="knapsack floor quality — keep at the shipped flat q so "
                         "no image can regress; lower only if the floor overruns")
    ap.add_argument("--qmax", type=int, default=98, help="knapsack ceiling quality")
    ap.add_argument("--qstep", type=int, default=1)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = REPO / "submissions/phase1" / args.out
    stage = out_root / "renders"
    if stage.exists():
        shutil.rmtree(stage)

    # 1) produce float-quality PILs per scene (v2 preferred, banked chain else)
    per_scene = {}
    n_v2 = 0
    for s in PUBLIC + PRIVATE:
        imgs = png_renders(s, args.suffix)  # lossless Kaggle/local --png output
        if imgs is None:
            imgs = apply_v2(s, args.suffix, args.variant, device)
        # big-backbone variant cache is a superset candidate: try it too
        if imgs is None:
            imgs = apply_v2(s, args.suffix, args.variant + "_bb", device)
        if imgs is not None:
            n_v2 += 1
            print(f"  {s}: v2 refiner ({len(imgs)} views)")
        else:
            imgs = fallback_images(s)
        per_scene[s] = imgs

    # 2) budget-fit the PRIVATE set (the LB zip) at 4:4:4 under BUDGET_MB.
    if args.encode == "knapsack":
        # exp037: allocate the byte budget ACROSS all private images at once, so
        # busy frames (where JPEG costs real Score) buy quality from flat sky
        # frames that give it up for free. Strictly dominates one flat q for the
        # whole set: the flat q IS in the feasible set the greedy searches over.
        pooled = [(f"{s}/{name}", im) for s in PRIVATE for name, im in per_scene[s]]
        alloc = knap16.allocate(pooled, int(BUDGET_MB * 1e6),
                                qualities=range(args.qmin, args.qmax + 1, args.qstep),
                                subsampling=0, tune="ms-ssim")
        enc_priv = {s: {} for s in PRIVATE}
        for key, (data, _q) in alloc.items():
            s, name = key.split("/", 1)
            enc_priv[s][name] = data
        qs = sorted(q for _, q in alloc.values())
        q = f"knap-q{qs[0]}..{qs[-1]}"
        total = sum(len(b) for d in enc_priv.values() for b in d.values())
        print(f"  knapsack: private total {total/1e6:.1f} MB, q range {qs[0]}-{qs[-1]}")
        if total / 1e6 > BUDGET_MB:
            raise SystemExit(f"knapsack floor q{args.qmin} = {total/1e6:.1f} MB exceeds "
                             f"{BUDGET_MB} MB budget — lower --qmin")
    else:
        q = 95
        while q >= 90:
            total = 0
            enc_priv = {}
            for s in PRIVATE:
                enc_priv[s] = encode_all(per_scene[s], q)
                total += sum(len(b) for b in enc_priv[s].values())
            print(f"  q{q}_sub0: private total {total/1e6:.1f} MB")
            if total / 1e6 <= BUDGET_MB:
                break
            q -= 1
        else:
            raise SystemExit("could not fit budget even at q90")
        q = f"q{q}"

    # 3) stage: private at chosen q, public at q90 (ungraded on the LB; keeps
    #    any hypothetical full-zip variant small)
    for s in PUBLIC + PRIVATE:
        d = stage / s / "renders_test"
        d.mkdir(parents=True, exist_ok=True)
        if s in PRIVATE:
            for name, data in enc_priv[s].items():
                (d / name).write_bytes(data)
        else:
            for name, data in encode_all(per_scene[s], 90).items():
                (d / name).write_bytes(data)

    # 4) package via the validated packager
    subprocess.run([sys.executable, str(REPO / "src/package_submission.py"),
                    "--runs-dir", str(stage), "--scenes", *PRIVATE,
                    "--poses-root", str(RAW_PRIV),
                    "--out", str(out_root / "partial_private_set1.zip")], check=True)
    subprocess.run([sys.executable, str(REPO / "src/package_submission.py"),
                    "--runs-dir", str(stage), "--scenes", *PUBLIC,
                    "--poses-root", str(RAW_PUB),
                    "--out", str(out_root / "partial_public_set.zip")], check=True)
    final = out_root / "submission_round1.zip"
    with zipfile.ZipFile(final, "w", zipfile.ZIP_STORED) as zout:
        for part in ("partial_public_set.zip", "partial_private_set1.zip"):
            with zipfile.ZipFile(out_root / part) as zin:
                for info in zin.infolist():
                    zout.writestr(info, zin.read(info))

    pz = out_root / "partial_private_set1.zip"
    print(f"\n{n_v2}/13 scenes on v2. Private zip {pz} = "
          f"{pz.stat().st_size/1e6:.0f} MB @ {q}_sub0 ({args.encode})")
    print(f"Full zip {final} = {final.stat().st_size/1e6:.0f} MB")


if __name__ == "__main__":
    main()
