#!/usr/bin/env bash
# Match-test hold-out validation for a GT-less round-2 scene (2026-07-18).
#
# Produces the FIRST real competition-metric number on a scene with no test GT,
# by holding out train frames that mimic the test-pose distribution, retraining
# the backbone WITHOUT them, and scoring those hold-outs against their real
# photos. Faithful only because these scenes are dense-interpolative (test poses
# sit in 3-4 deg gaps, frac_uncovered=0) -- a held-out frame lands in the same
# kind of gap a test frame does. The ranking half of this was validated in
# exp019 (Spearman 1.0 vs real test GT); this adds the held-out-generalization
# half exp019 skipped.
#
# Usage: scripts/val_holdout_run.sh <SCENE> <SPLIT> <PHASE> [N_VAL]
#   e.g. scripts/val_holdout_run.sh bonsai all round2 25
#
# Prereqs (from a prior `phase_run.sh <SCENE> <SPLIT> <PHASE>`):
#   data/processed/<PHASE>/<SCENE>/colmap_train_only/   (val poses live here)
#   data/processed/<PHASE>/<SCENE>/train_staging_dense/ (dense cloud reused)
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airace
source configs/phase_locked.conf

SCENE=${1:?usage: val_holdout_run.sh <SCENE> <SPLIT> <PHASE> [N_VAL]}
SPLIT=${2:?usage: val_holdout_run.sh <SCENE> <SPLIT> <PHASE> [N_VAL]}
PHASE=${3:?usage: val_holdout_run.sh <SCENE> <SPLIT> <PHASE> [N_VAL]}
N_VAL=${4:-25}

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SCENE_DIR=data/raw/$PHASE/$SPLIT/$SCENE
[ -d "$SCENE_DIR" ] || SCENE_DIR=data/raw/VAI_NVS_DATA_ROUND2/$SCENE   # round-2 raw root
SCENE_ROOT=data/processed/$PHASE/$SCENE
SPLIT_DIR=$SCENE_ROOT/splits
STAGING=$SCENE_ROOT/train_staging_holdout
RUN_DIR=runs/$PHASE/val_holdout/$SCENE

echo "== [1/4] match-test split (n_val=$N_VAL) =="
python src/data_prep/make_val_split.py --mode match-test \
  --images-dir "$SCENE_DIR/train/images" \
  --sparse-dir "$SCENE_ROOT/colmap_train_only" \
  --test-poses "$SCENE_DIR/test/test_poses.csv" \
  --out-dir "$SPLIT_DIR" --n-val "$N_VAL"

echo "== [2/4] hold-out staging (excludes val frames) =="
python src/data_prep/make_holdout_staging.py \
  --src-staging "$SCENE_ROOT/train_staging_dense" \
  --split-dir "$SPLIT_DIR" \
  --out-staging "$STAGING"

echo "== [3/4] retrain backbone on train-minus-holdout =="
final_ckpt=$(find "$RUN_DIR" -name "step-*.ckpt" 2>/dev/null | sort | tail -1 || true)
final_step=$(echo "$final_ckpt" | grep -oE '[0-9]+' | tail -1 || true)
if [ -n "$final_step" ] && [ "$((10#$final_step))" -ge "$((MAX_ITERS - 1))" ]; then
  echo "== backbone already complete (step $final_step) =="
else
  [ -d "$RUN_DIR" ] && { echo "== removing incomplete run =="; rm -rf "$RUN_DIR"; }
  # shellcheck disable=SC2086
  ns-train "$METHOD" \
    --data "$STAGING" \
    --output-dir "$RUN_DIR" \
    --max-num-iterations "$MAX_ITERS" \
    --viewer.quit-on-train-completion True \
    $EXTRA_ARGS \
    colmap --eval-mode all --colmap-path sparse/0 --downscale-factor 1
fi

echo "== [4/4] render val poses + score vs real photos (vgg+50) =="
config=$(find "$RUN_DIR" -name config.yml | sort | tail -1)
python src/render_val.py --config "$config" \
  --scene-dir "$SCENE_DIR" --processed-root "data/processed/$PHASE" \
  --out "$RUN_DIR/renders_val_split" \
  --metrics-out "$RUN_DIR/metrics_val_split.json"

echo "== VAL-HOLDOUT DONE: $SCENE -> $RUN_DIR/metrics_val_split.json =="
