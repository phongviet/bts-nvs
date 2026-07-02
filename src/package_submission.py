"""Assemble a submission ZIP from per-scene renders_test/ directories.

NOTE (Week 1, Day 5): the exact required ZIP layout/filenames must be confirmed
from the official rules doc and recorded in docs/rules_and_constraints.md
before trusting this script for a real submission. This implements the most
common convention (one folder per scene, exact filename casing preserved,
one image per required test pose) -- adjust ROOT_LAYOUT below once confirmed.
"""
import argparse
import shutil
import zipfile
from pathlib import Path


def package(runs_root: Path, scenes: list[str], out_zip: Path, render_subdir: str = "renders_test"):
    staging = out_zip.parent / (out_zip.stem + "_staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    missing = []
    for scene in scenes:
        src = runs_root / scene / render_subdir
        if not src.exists():
            missing.append(scene)
            continue
        dst = staging / scene
        shutil.copytree(src, dst)

    if missing:
        raise RuntimeError(f"Missing renders_test for scenes: {missing}. "
                            f"Fix before packaging -- a scene-count mismatch voids scoring.")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(staging))

    shutil.rmtree(staging)
    print(f"Wrote {out_zip} covering {len(scenes)} scenes.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, type=Path,
                     help="e.g. runs/phase1/exp001_baseline_splatfacto, containing <scene>/renders_test/")
    ap.add_argument("--scenes", required=True, nargs="+", help="exact scene folder names, case-sensitive")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    package(args.runs_dir, args.scenes, args.out)


if __name__ == "__main__":
    main()
