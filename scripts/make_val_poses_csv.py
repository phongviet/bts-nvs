"""Emit a render_test_csv-format pose CSV for a scene's match-test val frames.

The SSS renderer (sss_experiment/render_test_csv.py) consumes a CSV with columns
qw,qx,qy,qz,tx,ty,tz,width,height,fx,fy,image_name. The val-split poses live in
the scene's `colmap_train_only` model (which registers ALL frames; the split only
excludes val from the *training staging*) -- this is exactly the source
src/render_val.py uses for the splatfacto control, so SSS renders land in the same
world frame and are directly comparable to metrics_val_split.json.

Usage:
  python scripts/make_val_poses_csv.py --scene bonsai --phase round2 \
      --out sss_experiment/data/bonsai_val_poses.csv
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.render_utils import load_colmap_poses  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--phase", default="round2")
    ap.add_argument("--processed-root", default=None,
                    help="defaults to data/processed/<phase>")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    root = Path(args.processed_root or f"data/processed/{args.phase}") / args.scene
    val_ids = set((root / "splits" / "val_ids.txt").read_text().split())
    rows = load_colmap_poses(root / "colmap_train_only", only_names=val_ids)
    got = {r["image_name"] for r in rows}
    missing = val_ids - got
    if missing:
        raise SystemExit(f"{len(missing)} val poses not found in colmap_train_only: "
                         f"{sorted(missing)[:5]}...")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["qw", "qx", "qy", "qz", "tx", "ty", "tz",
            "width", "height", "fx", "fy", "image_name"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["image_name"]):
            qw, qx, qy, qz = r["qvec"]
            tx, ty, tz = r["tvec"]
            w.writerow({"qw": qw, "qx": qx, "qy": qy, "qz": qz,
                        "tx": tx, "ty": ty, "tz": tz,
                        "width": r["width"], "height": r["height"],
                        "fx": r["fx"], "fy": r["fy"],
                        "image_name": r["image_name"]})
    print(f"wrote {len(rows)} val poses -> {args.out}")


if __name__ == "__main__":
    main()
