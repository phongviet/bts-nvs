#!/usr/bin/env bash
# Week 2, backend-locking A/B test on ONE public scene (hcm0034), on top of
# the winning init from the init ablation (dense-COLMAP, arm b -- see
# results/week2_init_ablation.csv).
#
# Per docs/strategy.md Week 2 step 2: "Lock the backend on the winning init:
# splatfacto-big + MCMC (cap_max sweep), SH degree 3, 30k iters, Mip
# anti-aliasing, sky masking, tower scale regularization. A/B each feature."
#
# SCOPE NOTE: MCMC cap_max sweep, sky masking, and splatfacto-big are NOT
# included here.
#   - MCMC: this installed nerfstudio version's splatfacto.py hardcodes
#     gsplat.strategy.DefaultStrategy (no CLI switch to gsplat's
#     MCMCStrategy) -- swapping it needs a custom model subclass, out of
#     scope for a quick single-scene A/B. Flagging as a real follow-up if the
#     backend-locking phase gets more budget.
#   - Sky masking: no mask plumbing wired into the current dataparser/staging
#     dirs; needs a mask-generation step (e.g. SAM) not yet built.
#   - SH degree 3 is already nerfstudio's splatfacto default (verified in
#     installed models/splatfacto.py), so there's nothing to A/B there.
#   - splatfacto-big: tried on 2026-07-05, killed at 37% (11k/30k iters) --
#     its more aggressive densification (cull_alpha_thresh=0.005,
#     densify_grad_thresh=0.0005) grows the Gaussian count until it
#     saturates this 6GB card (5.87/6.14GB, 99% util) and iter time balloons
#     from ~93ms to ~700ms (ETA went from ~45min to >1 day). Deferred until
#     more VRAM is available -- not run in this pass.
#
# Knobs actually exposed on the CLI and tested here, each changed ALONE vs
# the exp002_dense_colmap_init/hcm0034 baseline already trained (arm b_dense,
# splatfacto, classic rasterize, no scale reg, sh_degree 3, score 0.7226):
#   1. antialiased: --pipeline.model.rasterize-mode antialiased (mip AA)
#   2. scale_reg  : --pipeline.model.use-scale-regularization True
#                   (PhysGaussian spike/tower regularization)
#   3. mcmc       : splatfacto-mcmc (src/models/splatfacto_mcmc.py + custom
#                   registration in src/register_custom_methods.py) --
#                   gsplat MCMCStrategy instead of DefaultStrategy, default
#                   cap_max=1_000_000. Smoke-tested 2026-07-05 (300 iters, no
#                   errors, stable ~98ms/iter) but NOT scored yet -- that's
#                   what running this script does.
#   4. sky_mask   : splatfacto + --masks-path masks, using sky masks from
#                   src/data_prep/build_sky_masks.py (generic ADE20K
#                   segformer-b0, ignores sky pixels in the loss). Requires
#                   the mask-generation step below to have been run first
#                   (already done for hcm0034 as of this script version).
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airace

SCENE=hcm0034
SPLIT=public_set
PROCESSED_ROOT=data/processed/phase1
RAW_ROOT=data/raw/phase1
OUT_ROOT=runs/phase1/exp004_backend_ablation
RESULTS_CSV=results/week2_backend_ablation.csv
MAX_ITERS=30000

STAGING="$PROCESSED_ROOT/$SCENE/train_staging_dense"
SCENE_DIR="$RAW_ROOT/$SPLIT/$SCENE"

# Needed for the "mcmc" variant -- registers splatfacto-mcmc via nerfstudio's
# env-var plugin mechanism (see src/register_custom_methods.py). Harmless for
# the other variants.
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export NERFSTUDIO_METHOD_CONFIGS="splatfacto-mcmc=src.register_custom_methods:splatfacto_mcmc_method"

if [ ! -d "$STAGING" ]; then
  echo "!! $STAGING missing -- build dense-COLMAP init for $SCENE first."
  exit 1
fi

if [ ! -d "$STAGING/masks" ]; then
  echo "== sky masks not built yet for $SCENE -- building now =="
  python src/data_prep/build_sky_masks.py \
    --scene-dir "$SCENE_DIR" --processed-root "$PROCESSED_ROOT" --staging-dir-name train_staging_dense
fi

if [ ! -f "$RESULTS_CSV" ]; then
  echo "variant,method,psnr,ssim,lpips,score,timestamp,git_commit" > "$RESULTS_CSV"
fi

append_row() {
  local variant=$1 method=$2 metrics_json=$3
  python - "$variant" "$method" "$metrics_json" "$RESULTS_CSV" <<'PYEOF'
import json, sys, datetime
variant, method, metrics_json, csv_path = sys.argv[1:5]
d = json.load(open(metrics_json))
m = d["mean"]
row = f"{variant},{method},{m['psnr']:.4f},{m['ssim']:.4f},{m['lpips']:.4f},{m['score']:.4f},{datetime.datetime.now().isoformat(timespec='seconds')},PENDING\n"
with open(csv_path, "a") as f:
    f.write(row)
print(row.strip())
PYEOF
}

run_variant() {
  # Usage: run_variant <variant> <method> [pipeline/model args...] [-- <colmap dataparser args...]]
  local variant=$1 method=$2; shift 2
  local pre_args=() post_args=() seen_sep=0
  for a in "$@"; do
    if [ "$a" = "--" ]; then seen_sep=1; continue; fi
    if [ "$seen_sep" -eq 0 ]; then pre_args+=("$a"); else post_args+=("$a"); fi
  done
  local run_dir="$OUT_ROOT/$variant"

  echo "=========================================="
  echo "== variant=$variant method=$method pre_args=${pre_args[*]:-none} post_args=${post_args[*]:-none} =="
  echo "=========================================="

  if [ -d "$run_dir" ]; then
    echo "== $variant: run_dir exists, skipping training =="
  else
    ns-train "$method" \
      --data "$STAGING" \
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
      --poses-csv "$SCENE_DIR/test/test_poses.csv" --out "$run_dir/renders_test"
    touch "$run_dir/renders_test/.done"
  fi

  python src/metrics.py --renders "$run_dir/renders_test" \
    --gt "$SCENE_DIR/test/images" --out "$run_dir/metrics_val.json"

  append_row "$variant" "$method" "$run_dir/metrics_val.json"
}

# Baseline for reference (already trained under exp002; copy its score into this CSV, don't retrain)
BASELINE_METRICS="runs/phase1/exp002_dense_colmap_init/$SCENE/metrics_val.json"
if [ -f "$BASELINE_METRICS" ] && ! grep -q "^baseline," "$RESULTS_CSV"; then
  append_row "baseline" "splatfacto" "$BASELINE_METRICS"
fi

# "big" variant deferred -- see SCOPE NOTE above. Re-add when more VRAM is available:
#   run_variant "big" "splatfacto-big"
run_variant "antialiased" "splatfacto" --pipeline.model.rasterize-mode antialiased
run_variant "scale_reg" "splatfacto" --pipeline.model.use-scale-regularization True
run_variant "mcmc" "splatfacto-mcmc"
run_variant "sky_mask" "splatfacto" -- --masks-path masks

echo "Backend A/B on $SCENE complete. Results in $RESULTS_CSV"
column -s, -t "$RESULTS_CSV"
