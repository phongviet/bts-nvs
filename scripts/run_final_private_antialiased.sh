#!/usr/bin/env bash
# Locked-in Week 2 config for the Round 1 graded submission (private_set1
# only -- public_set is local-validation-only, see docs/rules_and_constraints.md):
#   init    = dense-COLMAP (arm b) -- won the init ablation on all 5 public
#             scenes (+0.0037 mean score vs sparse control, see
#             results/week2_init_ablation.csv)
#   backend = splatfacto + --pipeline.model.rasterize-mode antialiased --
#             the only backend-locking feature that helped on every one of
#             the 4 scored public scenes (+0.0002 to +0.0028, mean +0.0016;
#             see results/week2_backend_ablation.csv). scale_reg and
#             sky_mask were both dropped (scale_reg: negative on all 4
#             scenes; sky_mask: ~0.0000 net effect, only ~14% of images
#             have any sky and it's a tiny fraction of the frame even then).
#   mcmc / splatfacto-big: deferred, this 6GB card can't fit a real
#             (non-no-op) cap_max or splatfacto-big's growth -- see
#             conversation / experiment_log.md.
#
# Trains + renders all 8 private_set1 scenes (no local GT for these, so no
# local scoring step -- that's expected, see docs/rules_and_constraints.md),
# then packages submission_round1.zip the same way as before.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airace

SCENES=(HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437)
SPLIT=private_set1
PROCESSED_ROOT=data/processed/phase1
RAW_ROOT=data/raw/phase1
OUT_ROOT=runs/phase1/exp005_antialiased_dense
MAX_ITERS=30000
# NOTE: each config gets its own folder -- exp002's zip here is the scored
# leaderboard datapoint #2 artifact; never overwrite a submitted zip.
SUBMISSION_ZIP=submissions/phase1/exp005_antialiased_dense_results/submission_round1.zip

for scene in "${SCENES[@]}"; do
  staging="$PROCESSED_ROOT/$scene/train_staging_dense"
  scene_dir="$RAW_ROOT/$SPLIT/$scene"
  run_dir="$OUT_ROOT/$scene"

  if [ ! -d "$staging" ]; then
    echo "!! $scene: $staging missing -- build dense-COLMAP init first. Skipping."
    continue
  fi

  echo "=========================================="
  echo "== $scene (dense-COLMAP init + antialiased) =="
  echo "=========================================="

  final_ckpt=$(find "$run_dir" -name "step-*.ckpt" 2>/dev/null | sort | tail -1)
  final_step=$(echo "$final_ckpt" | grep -oE '[0-9]+' | tail -1)

  if [ -n "$final_step" ] && [ "$((10#$final_step))" -ge "$((MAX_ITERS - 1))" ]; then
    echo "== $scene: training already complete (step $final_step), skipping =="
  else
    if [ -d "$run_dir" ]; then
      echo "== $scene: incomplete run found (last step: ${final_step:-none}), removing and retraining from scratch =="
      rm -rf "$run_dir"
    fi
    ns-train splatfacto \
      --data "$staging" \
      --output-dir "$run_dir" \
      --max-num-iterations "$MAX_ITERS" \
      --viewer.quit-on-train-completion True \
      --pipeline.model.rasterize-mode antialiased \
      colmap --eval-mode all --colmap-path sparse/0
  fi

  config=$(find "$run_dir" -name config.yml | sort | tail -1)

  if [ ! -f "$run_dir/renders_test/.done" ]; then
    python src/render.py --config "$config" --mode test \
      --poses-csv "$scene_dir/test/test_poses.csv" --out "$run_dir/renders_test"
    touch "$run_dir/renders_test/.done"
  fi

  echo "== $scene done =="
done

echo "All private scenes trained. Verifying render completeness before packaging..."
python3 - "$OUT_ROOT" "$RAW_ROOT/$SPLIT" "${SCENES[@]}" <<'PYEOF'
import csv, sys
from pathlib import Path
from PIL import Image

out_root, raw_split = Path(sys.argv[1]), Path(sys.argv[2])
scenes = sys.argv[3:]
all_ok = True
for scene in scenes:
    csv_path = raw_split / scene / "test" / "test_poses.csv"
    render_dir = out_root / scene / "renders_test"
    expected = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            expected[r["image_name"]] = (int(r["width"]), int(r["height"]))
    rendered = {p.name for p in render_dir.iterdir() if p.name != ".done"} if render_dir.exists() else set()
    missing = set(expected) - rendered
    size_mismatch = []
    for name, (w, h) in expected.items():
        p = render_dir / name
        if p.exists():
            iw, ih = Image.open(p).size
            if (iw, ih) != (w, h):
                size_mismatch.append(name)
    ok = not missing and not size_mismatch
    all_ok &= ok
    print(f"{scene}: {'OK' if ok else 'FAIL'} missing={len(missing)} size_mismatch={len(size_mismatch)}")
if not all_ok:
    print("FAILED verification -- not packaging. Fix the scene(s) above first.")
    sys.exit(1)
print("All scenes verified OK.")
PYEOF

echo "Packaging $SUBMISSION_ZIP ..."
python src/package_submission.py \
  --runs-dir "$OUT_ROOT" \
  --scenes "${SCENES[@]}" \
  --out "$SUBMISSION_ZIP"

echo "Done. $SUBMISSION_ZIP ready to upload."
