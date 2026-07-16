#!/usr/bin/env bash
# Phase-2/3 per-scene automation: raw scene dir -> test renders, no hand-state.
# (plan_execution_v3 W4 Thu-Fri: `make phase-run`, target <=2h/scene, parallel
# across scenes/GPUs -- run one invocation per scene per GPU.)
#
# Usage:
#   scripts/phase_run.sh <SCENE> <SPLIT> [PHASE]
#   e.g. scripts/phase_run.sh HCM0249 private_set1 phase1
#
# Stages (all resumable -- each is skipped if its output already exists):
#   1. filter COLMAP to train-only images     (filter_colmap_train.py)
#   2. DAY-1 REGIME TRIPWIRE: test-pose coverage -> interpolative/extrapolative
#      verdict. An extrapolative verdict does NOT stop the run, but it is
#      loudly flagged: it invalidates the interpolation-regime strategy and
#      must go to the team sync immediately (see docs/phase_runbook.md).
#   3. dense-COLMAP init                      (build_dense_colmap.py)
#   4. optional transient masks               (build_transient_masks.py)
#   5. train the locked config                (configs/phase_locked.conf)
#   6. render test poses                      (src/render.py --mode test)
# Packaging across scenes is a separate step: `make phase-package`.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airace

SCENE=${1:?usage: phase_run.sh <SCENE> <SPLIT> [PHASE]}
SPLIT=${2:?usage: phase_run.sh <SCENE> <SPLIT> [PHASE]}
PHASE=${3:-phase1}

source configs/phase_locked.conf

RAW_ROOT=data/raw/$PHASE
PROCESSED_ROOT=data/processed/$PHASE
SCENE_DIR=$RAW_ROOT/$SPLIT/$SCENE
RUN_DIR=runs/$PHASE/phase_locked/$SCENE
SCENE_ROOT=$PROCESSED_ROOT/$SCENE

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export NERFSTUDIO_METHOD_CONFIGS="\
splatfacto-mcmc=src.register_custom_methods:splatfacto_mcmc_method,\
splatfacto-perceptual=src.register_custom_methods:splatfacto_perceptual_method,\
splatfacto-mcmc-perceptual=src.register_custom_methods:splatfacto_mcmc_perceptual_method,\
splatfacto-tpw=src.register_custom_methods:splatfacto_tpw_method"

[ -d "$SCENE_DIR/train/images" ] || { echo "!! $SCENE_DIR/train/images missing"; exit 1; }
[ -f "$SCENE_DIR/test/test_poses.csv" ] || { echo "!! $SCENE_DIR/test/test_poses.csv missing"; exit 1; }
t_start=$(date +%s)

# --- 1. filter COLMAP to train-only ---
if [ ! -d "$SCENE_ROOT/colmap_train_only" ]; then
  python src/data_prep/filter_colmap_train.py \
    --scene-dir "$SCENE_DIR" --processed-root "$PROCESSED_ROOT"
fi

# --- 2. day-1 regime tripwire ---
python src/data_prep/test_pose_coverage.py \
  --raw-root "$RAW_ROOT" --processed-root "$PROCESSED_ROOT" --scenes "$SCENE" \
  --out-csv "results/${PHASE}_test_pose_coverage.csv" \
  --plots-dir "results/plots/$PHASE"
verdict=$(python - "$SCENE" "results/${PHASE}_test_pose_coverage_summary.csv" <<'PYEOF'
import csv, sys
scene, path = sys.argv[1:3]
with open(path) as f:
    for r in csv.DictReader(f):
        if r["scene"] == scene:
            print(r["regime"]); break
PYEOF
)
if [ "$verdict" != "interpolative" ]; then
  echo "##########################################################"
  echo "## REGIME TRIPWIRE: $SCENE verdict = '$verdict'"
  echo "## The interpolation-regime strategy does NOT hold here."
  echo "## Escalate to the team sync NOW (docs/phase_runbook.md §Tripwire)."
  echo "##########################################################"
fi

# --- 3. dense init ---
if [ ! -d "$SCENE_ROOT/train_staging_dense" ]; then
  python src/data_prep/build_dense_colmap.py \
    --scene-dir "$SCENE_DIR" --processed-root "$PROCESSED_ROOT" \
    --max-points "$DENSE_MAX_POINTS" --max-image-size "$DENSE_MAX_IMAGE_SIZE" --gpu-index 0
fi

# --- 4. optional transient masks ---
if [ "$USE_TRANSIENT_MASKS" = "1" ] && [ ! -d "$SCENE_ROOT/transient_masks" ]; then
  python src/data_prep/build_transient_masks.py \
    --images "$SCENE_DIR/train/images" --processed-root "$PROCESSED_ROOT" \
    --scene "$SCENE" --staging-dir-name train_staging_dense --qa
fi

# --- 5. train locked config ---
final_ckpt=$(find "$RUN_DIR" -name "step-*.ckpt" 2>/dev/null | sort | tail -1 || true)
# `|| true`: on a FRESH scene the ckpt list is empty and grep's exit-1 would
# kill the whole run under set -e/pipefail (bit us on round-2 chair, Jul-17)
final_step=$(echo "$final_ckpt" | grep -oE '[0-9]+' | tail -1 || true)
if [ -n "$final_step" ] && [ "$((10#$final_step))" -ge "$((MAX_ITERS - 1))" ]; then
  echo "== $SCENE: training complete (step $final_step) =="
else
  [ -d "$RUN_DIR" ] && { echo "== $SCENE: removing incomplete run =="; rm -rf "$RUN_DIR"; }
  # shellcheck disable=SC2086
  ns-train "$METHOD" \
    --data "$SCENE_ROOT/train_staging_dense" \
    --output-dir "$RUN_DIR" \
    --max-num-iterations "$MAX_ITERS" \
    --viewer.quit-on-train-completion True \
    $EXTRA_ARGS \
    colmap --eval-mode all --colmap-path sparse/0
fi

# --- 6. render test poses ---
config=$(find "$RUN_DIR" -name config.yml | sort | tail -1)
python src/render.py --config "$config" --mode test \
  --poses-csv "$SCENE_DIR/test/test_poses.csv" --out "$RUN_DIR/renders_test"

elapsed=$(( ($(date +%s) - t_start) / 60 ))
echo "== $SCENE done in ${elapsed} min (regime: $verdict) -> $RUN_DIR/renders_test =="
