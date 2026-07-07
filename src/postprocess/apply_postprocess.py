"""Apply the exp011 winning (op, encoder) to a renders dir -- the packaging
path. Filenames are preserved (the submission requires exact image_name
matches). Refuse the 'png' pseudo-encoder here: PNG bytes in .JPG-named
files is undefined scorer behavior.

Usage:
  python src/postprocess/apply_postprocess.py \
      --src runs/.../renders_test --dst runs/.../renders_test_pp \
      --op identity --encoder jpeg95
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.postprocess.ops import ENCODERS, OPS, process_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--op", required=True, choices=sorted(OPS.keys()))
    ap.add_argument("--encoder", required=True, choices=sorted(ENCODERS.keys()))
    args = ap.parse_args()

    if args.encoder == "png":
        raise SystemExit("Refusing 'png' in the packaging path: submissions keep .JPG names; "
                         "png is a local what-if encoder only.")
    args.dst.mkdir(parents=True, exist_ok=True)
    n = process_dir(args.src, args.dst, args.op, args.encoder)
    print(f"{args.src} -> {args.dst}: {n} images, op={args.op}, encoder={args.encoder}")


if __name__ == "__main__":
    main()
