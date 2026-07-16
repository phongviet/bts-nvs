"""Analysis 18 (exp041, Wave-2): import externally-rendered depth (RaDe-GS,
PGSR, 2DGS) into the DIBR Warper, and VALIDATE it against our 3DGS depth before
anything is A/B'd on it.

Why depth-only: our DIBR occlusion z-test is the single place a better geometry
model pays off directly, and it is the cheapest possible integration — we do NOT
adopt their renderer, their densification, or their appearance model, only the
per-train-view depth map used to decide "is this real pixel visible here?".
Everything else (fallback RGB, blending, guard, refiner) is unchanged, so the
A/B isolates depth quality. RaDe-GS (arXiv 2406.01467) reports markedly better
depth than vanilla 3DGS at similar render quality, which is exactly the axis the
z-test cares about.

Flow:
  1. Train RaDe-GS/PGSR in THEIR repo on OUR COLMAP + OUR train images (their
     licenses are non-commercial-research; we run their code standalone and
     import only the depth ARRAY, never their code into ours).
  2. Export per-train-view depth to {image_name}.npy, metres, our pinhole
     intrinsics, HxW = the train image size.
  3. `--validate` here: compares against our 3DGS depth cache and REFUSES the
     import unless the conventions line up (see check_depth for what "line up"
     means and why each check exists).
  4. Run the DIBR with `--depth-source <dir>`, keeping every other flag equal.

Convention we require (matching Warper.render / synthesize):
  * z-DEPTH along the camera's forward axis, NOT ray distance from the centre.
    (ray distance is ~1-3% larger off-axis and biases the z-test at the edges,
    which is exactly where thin structures live)
  * metres, in the SAME world scale as our COLMAP/nerfstudio poses
  * float32/float16 (H, W), pinhole (undistorted) geometry, no SIMPLE_RADIAL
  * 0 (or negative) = no depth / sky

Run:
  conda run -n airace python Analysis/18_import_depth.py --scene HCM0181 \
      --src /path/to/radegs/export --validate
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "Analysis/X3_dibr"


def _dibr():
    spec = importlib.util.spec_from_file_location("dibr04", REPO / "Analysis/04_x3_dibr_pilot.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check_depth(ours: np.ndarray, theirs: np.ndarray, name="") -> list[str]:
    """Returns a list of problems (empty = usable). Pure function -> unit-tested
    without a checkpoint load. Every check here corresponds to a real way an
    imported depth map silently degrades the z-test rather than failing."""
    problems = []
    if theirs.shape != ours.shape:
        problems.append(f"{name}: shape {theirs.shape} != ours {ours.shape}")
        return problems  # nothing else is meaningful
    if not np.isfinite(theirs).all():
        problems.append(f"{name}: contains NaN/Inf")
    valid = (ours > 0) & (theirs > 0)
    if valid.sum() < 0.2 * ours.size:
        problems.append(f"{name}: only {valid.mean()*100:.0f}% co-valid pixels")
        return problems
    o, t = ours[valid], theirs[valid]
    # 1) SCALE. A different world scale (their repo re-normalises the COLMAP
    #    scene) makes every z-test compare metres to arbitrary units.
    ratio = float(np.median(t / o))
    if not (0.9 < ratio < 1.1):
        problems.append(f"{name}: median depth ratio {ratio:.3f} — scale mismatch "
                        f"(expect ~1.0; renormalise or re-export in our scale)")
    # 2) SIGN/DIRECTION. Some exporters write disparity (1/z) or negated z.
    if np.corrcoef(o, t)[0, 1] < 0.5:
        problems.append(f"{name}: correlation with our depth {np.corrcoef(o,t)[0,1]:.2f} "
                        f"— check for disparity (1/z) or a flipped sign")
    # 3) RAY-DISTANCE vs Z-DEPTH. Distance-from-centre grows off-axis; if the
    #    ratio t/o trends outward from the principal point, they exported range.
    H, W = ours.shape
    gy, gx = np.mgrid[0:H, 0:W]
    r = np.hypot(gx - W / 2, gy - H / 2)[valid]
    rel = (t / o) / ratio
    if np.corrcoef(r, rel)[0, 1] > 0.3:
        problems.append(f"{name}: t/o grows with radius — looks like RAY DISTANCE, "
                        f"not z-depth along the forward axis")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--src", required=True, help="dir of {image_name}.npy depth exports")
    ap.add_argument("--dst", default=None, help="default Analysis/X3_dibr/{scene}/depth_import")
    ap.add_argument("--validate", action="store_true",
                    help="compare against our 3DGS depth cache before importing "
                         "(needs the cache; run the DIBR once first)")
    ap.add_argument("--n-check", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="import despite validation problems")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst) if args.dst else OUT / args.scene / "depth_import"
    npys = sorted(src.glob("*.npy"))
    if not npys:
        raise SystemExit(f"no *.npy under {src}")

    if args.validate:
        w = _dibr().Warper(args.scene)
        problems = []
        idxs = np.linspace(0, len(w.train) - 1, args.n_check).round().astype(int)
        for i in idxs:
            name = w.train[i][0]
            fp = src / (name + ".npy")
            if not fp.exists():
                problems.append(f"{name}: missing from {src}")
                continue
            problems += check_depth(w.train_depth(i).astype(np.float32),
                                    np.load(fp).astype(np.float32), name)
        if problems:
            print(f"VALIDATION: {len(problems)} problem(s) on {args.n_check} views:")
            for p in problems:
                print(f"  - {p}")
            if not args.force:
                raise SystemExit("refusing to import — fix the export or pass --force")
        else:
            print(f"VALIDATION OK on {args.n_check} views "
                  f"(scale, sign, and z-vs-range conventions all match)")

    dst.mkdir(parents=True, exist_ok=True)
    for f in npys:
        shutil.copy2(f, dst / f.name)
    print(f"imported {len(npys)} depth maps -> {dst}")
    print(f"A/B it with:\n  python Analysis/04_x3_dibr_pilot.py --scene {args.scene} "
          f"--mode traincheck --guard 0.18 --depth-source {dst}")


if __name__ == "__main__":
    main()
