"""Analysis 06: build the distortion-remapped 13-scene submission (exp030).

Applies Finding F1 (test GT is raw SIMPLE_RADIAL; see 05_camera_distortion_
findings.md) to the EXISTING locked-config fleet renders — no retraining.
Measured on hcm0034 this is +0.0544 Score; the two k=-0.115 HNI scenes should
gain far more. Output goes to submissions/phase1/exp030_distortion_remap_results/
as scene dirs + a validated submission_round1.zip built with the repo packager.

Run: conda run -n airace python Analysis/06_build_remapped_submission.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "phase1"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "Analysis"))

from importlib import import_module  # noqa: E402
x4 = import_module("03_x4_distortion_remap")
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary  # noqa: E402

RENDERS = {
    "hcm0034": "runs/phase1/exp004_backend_ablation/antialiased/renders_test",
    "hcm0031": "runs/phase1/exp004_backend_ablation/hcm0031_antialiased/renders_test",
    "HCM0181": "runs/phase1/exp004_backend_ablation/HCM0181_antialiased/renders_test",
    "HCM0193": "runs/phase1/exp004_backend_ablation/HCM0193_antialiased/renders_test",
    "HCM0204": "runs/phase1/exp004_hcm0204_fill/HCM0204/antialiased/renders_test",
    **{s: f"runs/phase1/exp005_antialiased_dense/{s}/renders_test"
       for s in ("HCM0249", "HCM0254", "HCM0276", "HCM1439",
                 "HNI0131", "HNI0265", "HNI0366", "HNI0437")},
}


def main():
    out_root = REPO / "submissions/phase1/exp030_distortion_remap_results"
    stage = out_root / "renders"
    for scene, rel in RENDERS.items():
        split = "public_set" if (RAW / "public_set" / scene).exists() else "private_set1"
        cams = read_cameras_binary(RAW / split / scene / "train/sparse/0/cameras.bin")
        cam = list(cams.values())[0]
        f, cx, cy, k = cam.params
        src = REPO / rel
        dst = stage / scene / "renders_test"
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for rf in sorted(src.glob("*.JPG")):
            img = np.asarray(Image.open(rf).convert("RGB"))
            Image.fromarray(x4.distort_remap(img, f, f, cx, cy, k)).save(dst / rf.name, quality=98)
            n += 1
        print(f"{scene}: {n} images remapped (k={k:+.5f})")

    # package public + private with the repo validator (it checks size/decode/format)
    for split, scenes in [("public_set", ["hcm0031", "hcm0034", "HCM0181", "HCM0193", "HCM0204"]),
                          ("private_set1", ["HCM0249", "HCM0254", "HCM0276", "HCM1439",
                                            "HNI0131", "HNI0265", "HNI0366", "HNI0437"])]:
        cmd = [sys.executable, str(REPO / "src/package_submission.py"),
               "--runs-dir", str(stage), "--scenes", *scenes,
               "--poses-root", str(RAW / split),
               "--out", str(out_root / f"partial_{split}.zip")]
        subprocess.run(cmd, check=True)

    # merge the two validated partial zips into the single required zip
    import zipfile
    final = out_root / "submission_round1.zip"
    with zipfile.ZipFile(final, "w", zipfile.ZIP_STORED) as zout:
        for part in ("partial_public_set.zip", "partial_private_set1.zip"):
            with zipfile.ZipFile(out_root / part) as zin:
                for info in zin.infolist():
                    zout.writestr(info, zin.read(info))
    print(f"\nWrote {final} ({final.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
