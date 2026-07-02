"""Hold out every Nth train image as a local validation split (Mip-NeRF360
convention), for scenes where test GT is not available (private_set1).

Writes data/processed/phase1/<scene>/splits/{train_ids.txt,val_ids.txt}.
"""
import argparse
from pathlib import Path


def make_split(images_dir: Path, out_dir: Path, every_n: int = 8):
    names = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    val = names[::every_n]
    train = [n for n in names if n not in set(val)]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "val_ids.txt").write_text("\n".join(val) + "\n")
    (out_dir / "train_ids.txt").write_text("\n".join(train) + "\n")
    print(f"{images_dir}: {len(train)} train / {len(val)} val (every {every_n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, type=Path, help="scene train/images dir")
    ap.add_argument("--out-dir", required=True, type=Path, help="scene splits output dir")
    ap.add_argument("--every-n", type=int, default=8)
    args = ap.parse_args()
    make_split(args.images_dir, args.out_dir, args.every_n)


if __name__ == "__main__":
    main()
