#!/usr/bin/env bash
# Backbone-side arms on chair (Analysis/PLAN_backbone_side_2026-07-26.md §3).
# Single-variable vs the existing control: plain splatfacto, ColmapDataParser, 30k,
# downscale 1, same train_staging_holdout (25 val frames withheld).
#   control raw-render hold-out Score = 0.6506 (PSNR 23.7653 SSIM 0.7688 LPIPS 0.3065)
#
# Usage: scripts/run_E1_E2_chair_backbone.sh <arm>
#   arm = E1  -> splatfacto-perceptual (LPIPS(VGG) loss, 512 crop)
#   arm = E2  -> splatfacto-depth      (scale-shift-invariant Depth-Anything-V2 loss)
set -euo pipefail
cd "$(dirname "$0")/.."
ARM="${1:?usage: $0 E1|E2}"
DP_EXTRA=()
RUNS=runs/round2/backbone_side
SCENE=chair
STAGING=data/processed/round2/$SCENE/train_staging_holdout

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export NERFSTUDIO_METHOD_CONFIGS="\
splatfacto-perceptual=src.register_custom_methods:splatfacto_perceptual_method,\
splatfacto-depth=src.register_custom_methods:splatfacto_depth_method"

case "$ARM" in
  E1) METHOD=splatfacto-perceptual
      # trace-every is NOT optional: exp009 was voided by a silent no-op.
      EXTRA=(--pipeline.model.lpips-loss-weight 0.1 --pipeline.model.lpips-trace-every 2000) ;;
  E2b) METHOD=splatfacto-depth
      EXTRA=(--pipeline.model.depth-loss-weight 0.1 --pipeline.model.depth-trace-every 2000)
      # Depth maps ride the dataparser, NOT a model arg: that is what keeps depth_filenames
      # index-aligned with image_filenames (see splatfacto_depth.py docstring).
      DP_EXTRA=(--depths-path mono_depth) ;;
  *)  echo "unknown arm $ARM" >&2; exit 2 ;;
esac

OUT="$RUNS/$ARM"
echo "=== $ARM: $METHOD on $SCENE (30k) -> $OUT ==="
conda run --no-capture-output -n airace ns-train "$METHOD" \
  --data "$STAGING" --output-dir "$OUT" \
  --experiment-name "$SCENE" --timestamp run \
  --max-num-iterations 30000 --steps-per-save 10000 \
  --steps-per-eval-image 5000 --steps-per-eval-all-images 30000 \
  --viewer.quit-on-train-completion True --vis tensorboard \
  "${EXTRA[@]}" \
  colmap --eval-mode all --colmap-path sparse/0 --downscale-factor 1 "${DP_EXTRA[@]}"

CFG="$OUT/$SCENE/$METHOD/run/config.yml"
echo "=== $ARM: rendering the 25 hold-out poses + scoring vs the real photos ==="
conda run --no-capture-output -n airace python src/render_val.py \
  --config "$CFG" \
  --scene-dir data/raw/VAI_NVS_DATA_ROUND2/$SCENE \
  --processed-root data/processed/round2 \
  --out "$OUT/renders_val_split" \
  --metrics-out "$OUT/metrics_val_split.json"

echo "=== $ARM DONE. control Score 0.6506 (LPIPS 0.3065, PSNR 23.7653) ==="
conda run -n airace python -c "
import json; m=json.load(open('$OUT/metrics_val_split.json')); m=m.get('mean',m)
c={'score':0.6506,'lpips':0.3065,'psnr':23.7653,'ssim':0.7688}
print('$ARM  ' + '  '.join(f'{k} {m[k]:.4f} (d{m[k]-c[k]:+.4f})' for k in ('score','lpips','psnr','ssim')))
print('VERDICT:', 'WIN' if m['score']-c['score'] > 0.002 and m['lpips'] < c['lpips'] else 'no (bar: +0.002 Score AND LPIPS better)')
"
