#!/usr/bin/env bash
# Week 2 backend-locking A/B, extended from hcm0034 (scripts/run_backend_ablation_hcm0034.sh)
# to the other 4 public scenes, for more evidence before locking backend
# features. Only the 3 variants that fit comfortably on this 6GB card:
#   - antialiased : --pipeline.model.rasterize-mode antialiased
#   - scale_reg   : --pipeline.model.use-scale-regularization True
#   - sky_mask    : --masks-path masks (src/data_prep/build_sky_masks.py)
#
# NOT included here (same as hcm0034's script) -- deferred until more VRAM:
#   - mcmc  (splatfacto-mcmc): a real (non-no-op) cap_max needs ~2.5-3M
#     Gaussians, which pushed this 6GB card to 96% usage and into visible
#     thrashing (iter time 100ms -> 273ms) at cap_max=3M on hcm0034 -- see
#     conversation / experiment_log.md.
#   - splatfacto-big: killed earlier on hcm0034 at 37%, same VRAM ceiling.
#
# All on top of the winning init (dense-COLMAP, arm b) already built for
# every public scene (data/processed/phase1/<scene>/train_staging_dense).
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airace

SCENES=(hcm0031 HCM0181 HCM0193 HCM0204)
SPLIT=public_set
PROCESSED_ROOT=data/processed/phase1
RAW_ROOT=data/raw/phase1
OUT_ROOT=runs/phase1/exp004_backend_ablation
RESULTS_CSV=results/week2_backend_ablation.csv
MAX_ITERS=30000

if [ ! -f "$RESULTS_CSV" ]; then
  echo "scene,variant,method,psnr,ssim,lpips,score,timestamp,git_commit" > "$RESULTS_CSV"
fi

append_row() {
  local scene=$1 variant=$2 method=$3 metrics_json=$4
  python - "$scene" "$variant" "$method" "$metrics_json" "$RESULTS_CSV" <<'PYEOF'
import json, sys, datetime
scene, variant, method, metrics_json, csv_path = sys.argv[1:6]
d = json.load(open(metrics_json))
m = d["mean"]
row = f"{scene},{variant},{method},{m['psnr']:.4f},{m['ssim']:.4f},{m['lpips']:.4f},{m['score']:.4f},{datetime.datetime.now().isoformat(timespec='seconds')},PENDING\n"
with open(csv_path, "a") as f:
    f.write(row)
print(row.strip())
PYEOF
}

run_variant() {
  # Usage: run_variant <scene> <staging> <scene_dir> <variant> <method> [pipeline/model args...] [-- <colmap dataparser args...>]
  local scene=$1 staging=$2 scene_dir=$3 variant=$4 method=$5; shift 5
  local pre_args=() post_args=() seen_sep=0
  for a in "$@"; do
    if [ "$a" = "--" ]; then seen_sep=1; continue; fi
    if [ "$seen_sep" -eq 0 ]; then pre_args+=("$a"); else post_args+=("$a"); fi
  done
  local run_dir="$OUT_ROOT/${scene}_${variant}"

  echo "=========================================="
  echo "== scene=$scene variant=$variant method=$method pre_args=${pre_args[*]:-none} post_args=${post_args[*]:-none} =="
  echo "=========================================="

  if grep -q "^${scene},${variant}," "$RESULTS_CSV" 2>/dev/null; then
    echo "== $scene/$variant: already scored in $RESULTS_CSV, skipping entirely =="
    return
  fi

  if [ -d "$run_dir" ]; then
    echo "== $scene/$variant: run_dir exists, skipping training =="
  else
    ns-train "$method" \
      --data "$staging" \
      --output-dir "$run_dir" \
      --max-num-iterations "$MAX_ITERS" \
      --viewer.quit-on-train-completion True \
      "${pre_args[@]}" \
      colmap --eval-mode all --colmap-path sparse/0 "${post_args[@]}"
  fi

  local config
  config=$(find "$run_dir" -name config.yml | sort | tail -1)

  if [ ! -f "$run_dir/renders_test/.done" ]; then
    python src/render.py --config "$config" --mode test \
      --poses-csv "$scene_dir/test/test_poses.csv" --out "$run_dir/renders_test"
    touch "$run_dir/renders_test/.done"
  fi

  python src/metrics.py --renders "$run_dir/renders_test" \
    --gt "$scene_dir/test/images" --out "$run_dir/metrics_val.json"

  append_row "$scene" "$variant" "$method" "$run_dir/metrics_val.json"
}

for scene in "${SCENES[@]}"; do
  staging="$PROCESSED_ROOT/$scene/train_staging_dense"
  scene_dir="$RAW_ROOT/$SPLIT/$scene"

  if [ ! -d "$staging" ]; then
    echo "!! $scene: $staging missing -- build dense-COLMAP init first. Skipping scene entirely."
    continue
  fi

  if [ ! -d "$staging/masks" ]; then
    echo "== $scene: sky masks not built yet -- building now =="
    python src/data_prep/build_sky_masks.py \
      --scene-dir "$scene_dir" --processed-root "$PROCESSED_ROOT" --staging-dir-name train_staging_dense
  fi

  # Baseline for reference, if this scene already has an exp002 dense-COLMAP score -- don't retrain.
  BASELINE_METRICS="runs/phase1/exp002_dense_colmap_init/$scene/metrics_val.json"
  if [ -f "$BASELINE_METRICS" ] && ! grep -q "^${scene},baseline," "$RESULTS_CSV"; then
    append_row "$scene" "baseline" "splatfacto" "$BASELINE_METRICS"
  fi

  run_variant "$scene" "$staging" "$scene_dir" "antialiased" "splatfacto" --pipeline.model.rasterize-mode antialiased
  run_variant "$scene" "$staging" "$scene_dir" "scale_reg" "splatfacto" --pipeline.model.use-scale-regularization True
  run_variant "$scene" "$staging" "$scene_dir" "sky_mask" "splatfacto" -- --masks-path masks
done

echo "Backend A/B on remaining public scenes complete. Results in $RESULTS_CSV"
column -s, -t "$RESULTS_CSV"
