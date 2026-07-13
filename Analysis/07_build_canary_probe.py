"""Analysis 07: build the LB-composition canary probe submission.

Question it answers (see 00_leaderboard_arithmetic.md): does the leaderboard
average include the 5 public scenes (H-all13) or only the 8 private (H-priv8)?
This decides what the top-8 numbers mean (stuffed vs genuine) and what private
quality we must reach.

Method: take the exp030 remapped submission staging, Gaussian-blur ONE public
scene's renders (hcm0034, sigma 3), measure the local score drop D on that
scene vs GT, package a full 13-scene zip. After submitting:
  LB drop ~= D * (1/13)  -> H-all13 (public counted; leaders near-certainly stuffed)
  LB drop ~= 0           -> H-priv8 (leaders genuine at 0.75+)
Submit the probe, read the LB, then RE-SUBMIT the best clean zip the same day
(system keeps the last upload; 5/day budget).

Run: conda run -n airace python Analysis/07_build_canary_probe.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "phase1"
sys.path.insert(0, str(REPO))

from src.metrics import compute_metrics  # noqa: E402

CANARY_SCENE = "hcm0034"
SIGMA = 3.0


def main():
    src_root = REPO / "submissions/phase1/exp030_distortion_remap_results/renders"
    out_root = REPO / "submissions/phase1/exp032_canary_probe_results"
    stage = out_root / "renders"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(src_root, stage)

    cdir = stage / CANARY_SCENE / "renders_test"
    for rf in sorted(cdir.glob("*.JPG")):
        img = Image.open(rf).convert("RGB").filter(ImageFilter.GaussianBlur(SIGMA))
        img.save(rf, quality=98)
    print(f"blurred {CANARY_SCENE} (sigma={SIGMA})")

    base = compute_metrics(src_root / CANARY_SCENE / "renders_test",
                           RAW / "public_set" / CANARY_SCENE / "test/images", "vgg", 50.0)["mean"]
    blur = compute_metrics(cdir,
                           RAW / "public_set" / CANARY_SCENE / "test/images", "vgg", 50.0)["mean"]
    D = base["score"] - blur["score"]
    print(f"{CANARY_SCENE}: clean {base['score']:.4f} -> blurred {blur['score']:.4f} (D={D:.4f})")
    print(f"predicted LB drop if H-all13: {100*D/13:.2f} pts; if H-priv8: 0.00 pts")

    for split, scenes in [("public_set", ["hcm0031", "hcm0034", "HCM0181", "HCM0193", "HCM0204"]),
                          ("private_set1", ["HCM0249", "HCM0254", "HCM0276", "HCM1439",
                                            "HNI0131", "HNI0265", "HNI0366", "HNI0437"])]:
        subprocess.run([sys.executable, str(REPO / "src/package_submission.py"),
                        "--runs-dir", str(stage), "--scenes", *scenes,
                        "--poses-root", str(RAW / split),
                        "--out", str(out_root / f"partial_{split}.zip")], check=True)
    import zipfile
    final = out_root / "submission_round1.zip"
    with zipfile.ZipFile(final, "w", zipfile.ZIP_STORED) as zout:
        for part in ("partial_public_set.zip", "partial_private_set1.zip"):
            with zipfile.ZipFile(out_root / part) as zin:
                for info in zin.infolist():
                    zout.writestr(info, zin.read(info))
    print(f"Wrote {final}")


if __name__ == "__main__":
    main()
