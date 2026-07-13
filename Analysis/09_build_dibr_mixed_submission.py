"""Analysis 09: build the P1 private submission = DIBR on the 4 traincheck-in-band
private scenes + F1-remap on the other 4 (kept from exp030, which already scored 70.45).

DIBR scenes (traincheck >= public -0.14 dB floor): HCM0249, HCM0254, HCM0276, HNI0366.
Remap scenes (out-of-band / need expanded-canvas DIBR): HCM1439, HNI0437, HNI0131, HNI0265.
Public 5 scenes: DIBR renders (all beat F1-remap on the real test metric).

Assembles a fresh staging tree, then reuses src/package_submission.py + the same
two-part ZIP_STORED merge as script 08. Output:
  submissions/phase1/exp032_dibr_mixed_results/{partial_private_set1,submission_round1}.zip

Run: conda run -n airace python Analysis/09_build_dibr_mixed_submission.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW_PUB = REPO / "data/raw/phase1/public_set"
RAW_PRIV = REPO / "data/raw/phase1/private_set1"
EXP030 = REPO / "submissions/phase1/exp030_distortion_remap_results/renders"
DIBR = REPO / "Analysis/X3_dibr"
OUT_ROOT = REPO / "submissions/phase1/exp032_dibr_mixed_results"
STAGE = OUT_ROOT / "renders"

DIBR_PRIVATE = ["HCM0249", "HCM0254", "HCM0276", "HNI0366", "HNI0131"]
REMAP_PRIVATE = ["HCM1439", "HNI0437", "HNI0265"]
DIBR_PUBLIC = ["hcm0031", "hcm0034", "HCM0181", "HCM0193", "HCM0204"]


def stage_scene(scene: str, src_render_dir: Path):
    dst = STAGE / scene / "renders_test"
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_render_dir, dst)
    n = len(list(dst.glob("*.JPG"))) + len(list(dst.glob("*.jpg")))
    print(f"  {scene}: {n} images from {src_render_dir.relative_to(REPO)}")
    return n


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    print("Staging DIBR private scenes:")
    for s in DIBR_PRIVATE:
        stage_scene(s, DIBR / s / "renders_g0.18")
    print("Staging F1-remap private scenes (from exp030):")
    for s in REMAP_PRIVATE:
        stage_scene(s, EXP030 / s / "renders_test")
    print("Staging DIBR public scenes:")
    for s in DIBR_PUBLIC:
        stage_scene(s, DIBR / s / "renders_g0.18")

    # validate + build the two partials, then merge (ZIP_STORED, like script 08)
    subprocess.run([sys.executable, str(REPO / "src/package_submission.py"),
                    "--runs-dir", str(STAGE),
                    "--scenes", *DIBR_PRIVATE, *REMAP_PRIVATE,
                    "--poses-root", str(RAW_PRIV),
                    "--out", str(OUT_ROOT / "partial_private_set1.zip")], check=True)
    subprocess.run([sys.executable, str(REPO / "src/package_submission.py"),
                    "--runs-dir", str(STAGE),
                    "--scenes", *DIBR_PUBLIC,
                    "--poses-root", str(RAW_PUB),
                    "--out", str(OUT_ROOT / "partial_public_set.zip")], check=True)

    final = OUT_ROOT / "submission_round1.zip"
    with zipfile.ZipFile(final, "w", zipfile.ZIP_STORED) as zout:
        for part in ("partial_public_set.zip", "partial_private_set1.zip"):
            with zipfile.ZipFile(OUT_ROOT / part) as zin:
                for info in zin.infolist():
                    zout.writestr(info, zin.read(info))
    print(f"\nBuilt {final} ({final.stat().st_size/1e6:.0f} MB)")
    print(f"Private-only zip for submission: {OUT_ROOT / 'partial_private_set1.zip'} "
          f"({(OUT_ROOT / 'partial_private_set1.zip').stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
