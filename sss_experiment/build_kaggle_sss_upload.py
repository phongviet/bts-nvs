"""Build kaggle-upload-sss.zip for the SSS public-scene fleet notebook.

ALLOWLIST packaging (exp022 rsync-exclude bug lesson: copy exactly what the
notebook needs and assert every piece):

  kaggle_upload/
    code/sss_experiment/
      3D-student-splatting-and-scooping/   (patched: eval=False, NaN guard)
      make_undistorted_scene.py
      render_test_csv.py
      kaggle_sss_fleet.py
      metrics.py                            (copy of bts-nvs src/metrics.py)
    data/<scene>/train_images/*.JPG
    data/<scene>/sparse0/{cameras,images,points3D}.bin
    data/<scene>/test_poses.csv
    data/<scene>/test_gt/*.JPG              (public GT, scoring only)

Run: python build_kaggle_sss_upload.py
"""
import shutil
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BTS = HERE.parent / "bts-nvs"
RAW = BTS / "data/raw/phase1/public_set"
STAGE = HERE / "_stage_sss/kaggle_upload"
OUT_ZIP = HERE.parent / "kaggle/kaggle-upload-sss.zip"

SCENES = ["hcm0031", "hcm0034", "HCM0181", "HCM0193", "HCM0204"]

CODE = ["make_undistorted_scene.py", "render_test_csv.py", "kaggle_sss_fleet.py"]


def cp(src: Path, dst: Path):
    assert src.exists(), f"MISSING: {src}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    # copyfile (not copy2): source data files are read-only; don't propagate
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=shutil.copyfile,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                      ".git", "build", "*.egg-info"))
    else:
        shutil.copyfile(src, dst)


def main():
    if STAGE.parent.exists():
        subprocess.run(["chmod", "-R", "u+w", str(STAGE.parent)], check=True)
        shutil.rmtree(STAGE.parent)

    code_dst = STAGE / "code/sss_experiment"
    cp(HERE / "3D-student-splatting-and-scooping",
       code_dst / "3D-student-splatting-and-scooping")
    for f in CODE:
        cp(HERE / f, code_dst / f)
    cp(BTS / "src/metrics.py", code_dst / "metrics.py")

    for s in SCENES:
        cp(RAW / s / "train/images", STAGE / f"data/{s}/train_images")
        for b in ("cameras.bin", "images.bin", "points3D.bin"):
            cp(RAW / s / "train/sparse/0" / b, STAGE / f"data/{s}/sparse0" / b)
        cp(RAW / s / "test/test_poses.csv", STAGE / f"data/{s}/test_poses.csv")
        cp(RAW / s / "test/images", STAGE / f"data/{s}/test_gt")

    # load-bearing asserts (paths the notebook/driver hard-code)
    for p in ("code/sss_experiment/kaggle_sss_fleet.py",
              "code/sss_experiment/metrics.py",
              "code/sss_experiment/3D-student-splatting-and-scooping/train.py",
              "code/sss_experiment/3D-student-splatting-and-scooping/submodules/diff-t-rasterization/setup.py"):
        assert (STAGE / p).exists(), f"stage missing {p}"
    # the two local patches MUST be in the upload
    args_py = (STAGE / "code/sss_experiment/3D-student-splatting-and-scooping/"
               "arguments/__init__.py").read_text()
    assert "self.eval = False" in args_py, "eval=False patch missing"
    rast_py = (STAGE / "code/sss_experiment/3D-student-splatting-and-scooping/"
               "submodules/diff-t-rasterization/diff_t_rasterization/__init__.py").read_text()
    assert "NaN grads sanitized" in rast_py, "NaN-guard patch missing"
    for s in SCENES:
        for p in (f"data/{s}/train_images", f"data/{s}/sparse0/points3D.bin",
                  f"data/{s}/test_poses.csv", f"data/{s}/test_gt"):
            assert (STAGE / p).exists(), f"stage missing {p}"

    OUT_ZIP.parent.mkdir(exist_ok=True)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_STORED) as z:  # jpg/bin: no recompress
        for f in sorted(STAGE.parent.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(STAGE.parent))
    print(f"wrote {OUT_ZIP} ({OUT_ZIP.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
