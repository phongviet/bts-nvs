#!/usr/bin/env bash
# exp017 -- final frozen-config fleet run (plan_execution_v3 W4 Tue-Wed).
# Reruns EVERYTHING from committed configs, no hand-state: per-scene
# phase_run.sh (locked config) -> per-scene overrides from
# configs/scene_overrides/<scene>.yaml (enhancer stage from gate_enhancer.py,
# postprocess op/encoder from the exp011 winner) -> package + validate.
# This run's artifacts are the reproducibility submission.
#
# Usage:
#   scripts/run_final_fleet.sh                       # all 8 private scenes
#   scripts/run_final_fleet.sh HNI0265 HNI0366       # subset
# Env overrides: PHASE (phase1), SPLIT (private_set1),
#   POSTPROC_OP / POSTPROC_ENCODER (exp011 winner; default identity/jpeg95),
#   LORA_DIR (exp016 checkpoint, needed if any scene gates 'finetuned').
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PHASE=${PHASE:-phase1}
SPLIT=${SPLIT:-private_set1}
POSTPROC_OP=${POSTPROC_OP:-identity}
POSTPROC_ENCODER=${POSTPROC_ENCODER:-jpeg95}
LORA_DIR=${LORA_DIR:-runs/phase1/exp016_difix_lora/best}

DEFAULT_SCENES=(HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437)
if [ "$#" -gt 0 ]; then SCENES=("$@"); else SCENES=("${DEFAULT_SCENES[@]}"); fi

RUN_ROOT=runs/$PHASE/phase_locked
OUT_ROOT=runs/$PHASE/exp017_final_fleet

enhancer_of() {  # scene -> off|offtheshelf|finetuned (from the gate's YAML)
  local f="configs/scene_overrides/$1.yaml"
  [ -f "$f" ] && grep -E "^enhancer:" "$f" | awk '{print $2}' || echo off
}

for scene in "${SCENES[@]}"; do
  echo "=============== $scene ==============="
  # 1. train + render test poses with the locked config (resumable)
  scripts/phase_run.sh "$scene" "$SPLIT" "$PHASE"

  src="$RUN_ROOT/$scene/renders_test"
  stage=$(enhancer_of "$scene")

  # 2. per-scene enhancer stage (gate decision from gate_enhancer.py)
  if [ "$stage" = "offtheshelf" ]; then
    python src/enhancer/run_difix.py --src "$src" --dst "$OUT_ROOT/$scene/enhanced"
    src="$OUT_ROOT/$scene/enhanced"
  elif [ "$stage" = "finetuned" ]; then
    [ -d "$LORA_DIR" ] || { echo "!! $scene gated 'finetuned' but $LORA_DIR missing"; exit 1; }
    python src/enhancer/run_difix.py --src "$src" --dst "$OUT_ROOT/$scene/enhanced" --lora "$LORA_DIR"
    src="$OUT_ROOT/$scene/enhanced"
  fi

  # 3. postprocess winner (exp011) into the packaging layout
  python src/postprocess/apply_postprocess.py --src "$src" \
    --dst "$OUT_ROOT/$scene/renders_test" \
    --op "$POSTPROC_OP" --encoder "$POSTPROC_ENCODER"
  echo "== $scene: enhancer=$stage postproc=$POSTPROC_OP/$POSTPROC_ENCODER =="
done

# 4. package + validate (pre- and post-zip)
python src/package_submission.py \
  --runs-dir "$OUT_ROOT" \
  --scenes "${SCENES[@]}" \
  --poses-root "data/raw/$PHASE/$SPLIT" \
  --out "submissions/$PHASE/exp017_final_fleet_results/submission_round1.zip"
echo "exp017 fleet complete -> submissions/$PHASE/exp017_final_fleet_results/"
