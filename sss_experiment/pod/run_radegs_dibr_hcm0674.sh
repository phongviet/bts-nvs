#!/usr/bin/env bash
# HCM0674: splatfacto-big (proven fleet config, 30k) as render backbone, but the
# DIBR occlusion z-test fed RaDe-GS train-view depth instead of splatfacto's own.
# Clean A/B vs the fleet HCM0674 (which scored inside LB 74.83): EVERYTHING equal
# except --depth-source. Only the z-test depth changes; splatfacto still supplies
# rgb_T and target-pose depth_T.
#
#   usage: bash run_radegs_dibr_hcm0674.sh
set -uo pipefail
ROOT=${ROOT:-/workspace/rd_run}
BTS=$ROOT/bts-nvs
SCENE=HCM0674
STAGING=$BTS/data/processed/round2/$SCENE/train_staging_dense
RADEGS_TRAIN_DEPTH=$ROOT/radegs_train_depth      # 240 train-view .JPG.npy (raw COLMAP units)
RUN=$ROOT/out/$SCENE/splatbig                    # ns-train output dir
GPUCSV=$ROOT/out/$SCENE/gpu_metrics.csv
LOG=$ROOT/out/$SCENE/pipeline.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

export PATH=/opt/miniconda/bin:$PATH
source /opt/miniconda/etc/profile.d/conda.sh
conda activate airace
export CUDA_HOME=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
export PYTHONUNBUFFERED=1

echo "== [$SCENE] RaDe-GS-DIBR pipeline  [$(date)] =="
bash "$ROOT/gpu_monitor.sh" "$GPUCSV" 5 & MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT

# staging images -> raw train images (bundle ships sparse only)
[ -e "$STAGING/images" ] || ln -s "$BTS/data/raw/VAI_NVS_DATA_ROUND2/$SCENE/train/images" "$STAGING/images"

# --- Stage 1: splatfacto-big 30k (proven fleet config) ---
CKPT=$(ls "$RUN"/*/*/*/nerfstudio_models/step-*.ckpt 2>/dev/null | tail -1 || true)
if [ -z "$CKPT" ]; then
  echo "== Stage 1: ns-train splatfacto-big 30k [$(date +%H:%M)] =="
  ns-train splatfacto-big --data "$STAGING" --output-dir "$RUN" \
    --max-num-iterations 30000 --viewer.quit-on-train-completion True \
    --pipeline.model.rasterize-mode antialiased \
    colmap --eval-mode all --colmap-path sparse/0 --downscale-factor 1
fi
# config lives in the same run dir as the checkpoint (<ts>/nerfstudio_models/step-*.ckpt)
CKPT=$(ls "$RUN"/*/*/*/nerfstudio_models/step-*.ckpt 2>/dev/null | tail -1 || true)
CFG=$([ -n "$CKPT" ] && echo "$(dirname "$(dirname "$CKPT")")/config.yml" || echo "")
echo "== splatfacto config: $CFG =="
[ -f "$CFG" ] || { echo "FATAL: no config.yml"; exit 1; }

cd "$BTS/Analysis"

# --- Stage 2: import + validate RaDe-GS depth into splatfacto (nerfstudio) scale ---
echo "== Stage 2: import RaDe-GS depth (rescale-ns + validate) [$(date +%H:%M)] =="
python 18_import_depth.py --scene $SCENE --src "$RADEGS_TRAIN_DEPTH" \
  --config "$CFG" --rescale-ns --validate
IMPORTED=$BTS/Analysis/X3_dibr/$SCENE/depth_import
echo "   imported -> $IMPORTED ($(ls "$IMPORTED"/*.npy 2>/dev/null | wc -l) maps)"

# --- Stage 3: refiner, fleet config + ONLY --depth-source swapped ---
echo "== Stage 3: refiner (splatfacto backbone + RaDe-GS z-test depth) [$(date +%H:%M)] =="
python 10_refiner_pilot.py --scene $SCENE --config "$CFG" \
  --ss 2 --sample cubic --base 48 --iters 6000 --ema 0.999 --tta --png \
  --suffix _rd --max-pairs 90 --depth-source "$IMPORTED"

REF=$BTS/Analysis/X5_refiner/$SCENE
OUT=$ROOT/out/$SCENE
mkdir -p "$OUT/deliver"
cp -rf "$REF/renders_refined_rd" "$OUT/deliver/" 2>/dev/null || true
cp -f "$REF/refiner_rd.pt" "$OUT/deliver/" 2>/dev/null || true
cp -f "$CFG" "$OUT/deliver/splatfacto_config.yml" 2>/dev/null || true
cp -rf "$RUN"/*/*/nerfstudio_models "$OUT/deliver/nerfstudio_models" 2>/dev/null || true
kill $MON 2>/dev/null || true
echo "== [$SCENE] DONE. refined=$(ls "$OUT/deliver/renders_refined_rd" 2>/dev/null | wc -l) PNG =="
echo "== peak VRAM: $(awk -F, 'NR>1{gsub(/ /,"",$3); if($3+0>m)m=$3+0}END{print m" MiB"}' "$GPUCSV" 2>/dev/null) =="
echo "== PIPELINE COMPLETE [$(date)] =="
