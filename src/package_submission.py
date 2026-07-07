"""Assemble and validate a submission ZIP from per-scene renders_test/ dirs.

Layout (confirmed by two scored submissions, see SUBMISSION_LOG.md):
one folder per scene at the zip root, exact filename casing preserved, one
image per required test pose. Every upload must be NAMED submission_round1.zip
(rules page); submissions are identified by their exp folder path.

The validator hard-fails on: missing/extra scenes, missing/extra images,
width/height mismatch vs test_poses.csv, undecodable files, and non-JPEG
bytes in .JPG-named files. Run it standalone on any existing zip with
--validate-zip.

Usage:
  package:  python src/package_submission.py --runs-dir runs/phase1/exp005_antialiased_dense \
                --scenes HCM0249 ... --poses-root data/raw/phase1/private_set1 \
                --out submissions/phase1/<exp>_results/submission_round1.zip
  validate: python src/package_submission.py --validate-zip <zip> \
                --scenes HCM0249 ... --poses-root data/raw/phase1/private_set1
"""
import argparse
import csv
import io
import shutil
import zipfile
from pathlib import Path

from PIL import Image

FORMAT_FOR_SUFFIX = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}


def load_expected(poses_root: Path, scene: str) -> dict[str, tuple[int, int]]:
    csv_path = poses_root / scene / "test" / "test_poses.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} missing -- wrong --poses-root or scene name?")
    expected = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            expected[r["image_name"]] = (int(r["width"]), int(r["height"]))
    return expected


def _check_image_bytes(name: str, data: bytes, want_wh: tuple[int, int]) -> list[str]:
    errors = []
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # decodability
        img = Image.open(io.BytesIO(data))  # verify() invalidates the object
    except Exception as e:
        return [f"{name}: undecodable ({e})"]
    if img.size != want_wh:
        errors.append(f"{name}: size {img.size} != expected {want_wh}")
    want_format = FORMAT_FOR_SUFFIX.get(Path(name).suffix.lower())
    if want_format and img.format != want_format:
        errors.append(f"{name}: {img.format} bytes in a {Path(name).suffix} file")
    return errors


def validate_scene_files(files: dict[str, bytes], expected: dict[str, tuple[int, int]],
                         scene: str) -> list[str]:
    """files: {image_name: raw bytes} for one scene."""
    errors = []
    missing = sorted(set(expected) - set(files))
    extra = sorted(set(files) - set(expected))
    if missing:
        errors.append(f"{scene}: {len(missing)} missing images (first: {missing[:3]})")
    if extra:
        errors.append(f"{scene}: {len(extra)} extra files (first: {extra[:3]})")
    for name in sorted(set(files) & set(expected)):
        errors += [f"{scene}/{e}" for e in _check_image_bytes(name, files[name], expected[name])]
    return errors


def _read_dir_files(render_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(render_dir.iterdir())
            if p.is_file() and p.name != ".done"}


def validate_renders(runs_root: Path, scenes: list[str], poses_root: Path,
                     render_subdir: str = "renders_test") -> list[str]:
    errors = []
    for scene in scenes:
        render_dir = runs_root / scene / render_subdir
        if not render_dir.exists():
            errors.append(f"{scene}: {render_dir} missing")
            continue
        errors += validate_scene_files(_read_dir_files(render_dir),
                                       load_expected(poses_root, scene), scene)
    return errors


def validate_zip(zip_path: Path, scenes: list[str], poses_root: Path) -> list[str]:
    errors = []
    with zipfile.ZipFile(zip_path) as zf:
        by_scene: dict[str, dict[str, bytes]] = {}
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if len(parts) != 2:
                errors.append(f"unexpected zip entry (want <scene>/<image>): {info.filename}")
                continue
            by_scene.setdefault(parts[0], {})[parts[1]] = zf.read(info)
    zip_scenes = set(by_scene)
    missing = sorted(set(scenes) - zip_scenes)
    extra = sorted(zip_scenes - set(scenes))
    if missing:
        errors.append(f"zip missing scenes: {missing}")
    if extra:
        errors.append(f"zip has unexpected scenes: {extra}")
    for scene in sorted(set(scenes) & zip_scenes):
        errors += validate_scene_files(by_scene[scene], load_expected(poses_root, scene), scene)
    return errors


def package(runs_root: Path, scenes: list[str], out_zip: Path,
            render_subdir: str = "renders_test", poses_root: Path | None = None):
    if poses_root is not None:
        errors = validate_renders(runs_root, scenes, poses_root, render_subdir)
        if errors:
            for e in errors:
                print("VALIDATION:", e)
            raise RuntimeError(f"{len(errors)} validation error(s) -- not packaging. "
                               f"A name/size/count mismatch voids scoring.")
        print(f"Pre-package validation OK: {len(scenes)} scenes vs test_poses.csv.")
    else:
        print("WARNING: no --poses-root given -- packaging without name/size/count validation.")

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
        shutil.copytree(src, staging / scene, ignore=shutil.ignore_patterns(".done"))

    if missing:
        raise RuntimeError(f"Missing {render_subdir} for scenes: {missing}. "
                           f"Fix before packaging -- a scene-count mismatch voids scoring.")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(staging))
    shutil.rmtree(staging)

    if poses_root is not None:
        errors = validate_zip(out_zip, scenes, poses_root)
        if errors:
            for e in errors:
                print("ZIP VALIDATION:", e)
            raise RuntimeError("packaged zip failed validation -- do not upload.")
    size_mb = out_zip.stat().st_size / 2**20
    print(f"Wrote {out_zip} covering {len(scenes)} scenes ({size_mb:.1f} MB). "
          f"Reminder: upload it named submission_round1.zip.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path,
                    help="e.g. runs/phase1/exp005_antialiased_dense, containing <scene>/renders_test/")
    ap.add_argument("--scenes", required=True, nargs="+", help="exact scene folder names, case-sensitive")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--render-subdir", default="renders_test")
    ap.add_argument("--poses-root", type=Path,
                    help="e.g. data/raw/phase1/private_set1 -- enables full validation (recommended)")
    ap.add_argument("--validate-zip", type=Path, help="validate an existing zip instead of packaging")
    args = ap.parse_args()

    if args.validate_zip:
        if args.poses_root is None:
            ap.error("--validate-zip requires --poses-root")
        errors = validate_zip(args.validate_zip, args.scenes, args.poses_root)
        if errors:
            for e in errors:
                print("ZIP VALIDATION:", e)
            raise SystemExit(1)
        print(f"{args.validate_zip}: OK ({len(args.scenes)} scenes).")
        return

    if not (args.runs_dir and args.out):
        ap.error("packaging requires --runs-dir and --out")
    package(args.runs_dir, args.scenes, args.out, args.render_subdir, args.poses_root)


if __name__ == "__main__":
    main()
