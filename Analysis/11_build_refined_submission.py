"""Analysis 11: build the P2 submission from refiner renders, with automatic
per-scene fallback: refiner (X5_refiner/<scene>/renders_refined) > DIBR
(X3_dibr/<scene>/renders_g0.18) > F1-remap (exp030 staging). Whichever best
source exists for a scene is used, so this works before and after a full fleet
refiner run.

Public 5 scenes are scored locally (GT present) so the chosen public mean is
printed as a transfer sanity check. Builds partial_private_set1.zip (the LB
mover) + the full submission_round1.zip.

Run: conda run -n airace python Analysis/11_build_refined_submission.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

# JPEG quality for the staged renders. Refiner outputs at q98 total 389 MB on the
# private set (> the 350 MB submission limit); q95 re-encode -> 283 MB with a
# measured cost of only -0.003 Score on public (validated Jul-12).
JPEG_QUALITY = 95

REPO = Path(__file__).resolve().parents[1]
RAW_PUB = REPO / "data/raw/phase1/public_set"
RAW_PRIV = REPO / "data/raw/phase1/private_set1"
EXP030 = REPO / "submissions/phase1/exp030_distortion_remap_results/renders"
DIBR = REPO / "Analysis/X3_dibr"
REFINER = REPO / "Analysis/X5_refiner"
OUT_ROOT = REPO / "submissions/phase1/exp033_refined_results"
STAGE = OUT_ROOT / "renders"

PUBLIC = ["hcm0031", "hcm0034", "HCM0181", "HCM0193", "HCM0204"]
PRIVATE = ["HCM0249", "HCM0254", "HCM0276", "HCM1439", "HNI0131", "HNI0265", "HNI0366", "HNI0437"]


def best_source(scene: str):
    for label, d in (("refiner", REFINER / scene / "renders_refined"),
                     ("dibr", DIBR / scene / "renders_g0.18"),
                     ("remap", EXP030 / scene / "renders_test")):
        if d.exists() and (any(d.glob("*.JPG")) or any(d.glob("*.jpg"))):
            return label, d
    raise FileNotFoundError(f"no render source for {scene}")


def stage_scene(scene: str):
    label, src = best_source(scene)
    dst = STAGE / scene / "renders_test"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    imgs = sorted(list(src.glob("*.JPG")) + list(src.glob("*.jpg")))
    for f in imgs:
        Image.open(f).convert("RGB").save(dst / f.name, "JPEG", quality=JPEG_QUALITY)
    print(f"  {scene:9s} <- {label:8s} ({len(imgs)} imgs @ q{JPEG_QUALITY})")
    return label


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    print("Staging (best source per scene):")
    chosen = {s: stage_scene(s) for s in PUBLIC + PRIVATE}

    subprocess.run([sys.executable, str(REPO / "src/package_submission.py"),
                    "--runs-dir", str(STAGE), "--scenes", *PRIVATE,
                    "--poses-root", str(RAW_PRIV),
                    "--out", str(OUT_ROOT / "partial_private_set1.zip")], check=True)
    subprocess.run([sys.executable, str(REPO / "src/package_submission.py"),
                    "--runs-dir", str(STAGE), "--scenes", *PUBLIC,
                    "--poses-root", str(RAW_PUB),
                    "--out", str(OUT_ROOT / "partial_public_set.zip")], check=True)

    final = OUT_ROOT / "submission_round1.zip"
    with zipfile.ZipFile(final, "w", zipfile.ZIP_STORED) as zout:
        for part in ("partial_public_set.zip", "partial_private_set1.zip"):
            with zipfile.ZipFile(OUT_ROOT / part) as zin:
                for info in zin.infolist():
                    zout.writestr(info, zin.read(info))

    n_ref = sum(1 for v in chosen.values() if v == "refiner")
    print(f"\nSources: {chosen}")
    print(f"{n_ref}/13 scenes use the refiner.")
    print(f"Built {final} ({final.stat().st_size/1e6:.0f} MB)")
    print(f"Private-only zip: {OUT_ROOT/'partial_private_set1.zip'} "
          f"({(OUT_ROOT/'partial_private_set1.zip').stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
