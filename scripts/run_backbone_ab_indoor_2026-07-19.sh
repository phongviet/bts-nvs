#!/usr/bin/env bash
# Backbone A/B on the round-2 INDOOR scenes (2026-07-19): splatfacto-big vs the
# locked splatfacto, scored by match-test hold-out.
#
# Why this exists: every backbone decision on record was made on drone scenes.
# bonsai/chair are handheld indoor video -- denser interpolation, motion blur,
# glossy reflections -- and their backbone-only scores sit far below the drone
# scenes, almost entirely on LPIPS (0.31/0.31 vs ~0.12 refined on hcm0034):
#
#   bonsai  PSNR 25.619  SSIM 0.8382  LPIPS 0.3232  Score 0.6759
#   chair   PSNR 23.765  SSIM 0.7688  LPIPS 0.3065  Score 0.6506
#
# splatfacto-big is the cheapest lever that plausibly moves LPIPS (more
# gaussians -> finer high-frequency detail). This measures whether it does, on
# the ONLY two round-2 scenes we can currently score at all.
#
# Fairness: same train-minus-25 staging, same 25 val frames, same scorer as the
# references above. Only --method differs, so any delta is the backbone.
#
# VRAM note: splatfacto-big on a 6GB card at 1920x1080 (bonsai) is the tight
# case. If a scene OOMs, that is itself the finding -- big is not affordable
# here without downscaling, which would confound the comparison.
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PHASE=round2
METHOD=splatfacto-big
MAX_ITERS=30000
LOG=results/local_wave_jul19
mkdir -p "$LOG"

for SCENE in bonsai chair; do
  SCENE_DIR=data/raw/VAI_NVS_DATA_ROUND2/$SCENE
  STAGING=data/processed/$PHASE/$SCENE/train_staging_holdout
  RUN_DIR=runs/$PHASE/val_holdout_big/$SCENE
  MET=$RUN_DIR/metrics_val_split.json

  if [ -f "$MET" ]; then echo "== SKIP $SCENE (scored)"; continue; fi
  [ -d "$STAGING" ] || { echo "== SKIP $SCENE: no holdout staging at $STAGING"; continue; }

  echo "== [$SCENE] train $METHOD [$(date +%H:%M)] =="
  if ! ns-train "$METHOD" \
      --data "$STAGING" \
      --output-dir "$RUN_DIR" \
      --max-num-iterations "$MAX_ITERS" \
      --viewer.quit-on-train-completion True \
      --pipeline.model.rasterize-mode antialiased \
      colmap --eval-mode all --colmap-path sparse/0 --downscale-factor 1 \
      > "$LOG/30_${SCENE}_big.log" 2>&1; then
    echo "== [$SCENE] TRAIN FAILED -- see $LOG/30_${SCENE}_big.log"
    grep -iE "out of memory|CUDA error" "$LOG/30_${SCENE}_big.log" | tail -3
    continue
  fi

  echo "== [$SCENE] render + score val poses [$(date +%H:%M)] =="
  config=$(find "$RUN_DIR" -name config.yml | sort | tail -1)
  python src/render_val.py --config "$config" \
    --scene-dir "$SCENE_DIR" --processed-root "data/processed/$PHASE" \
    --out "$RUN_DIR/renders_val_split" --metrics-out "$MET" \
    >> "$LOG/30_${SCENE}_big.log" 2>&1
  echo "== [$SCENE] DONE [$(date +%H:%M)]:"; cat "$MET"
done

echo "== BACKBONE A/B COMPLETE [$(date)] =="
echo "Compare vs splatfacto: bonsai 0.6759, chair 0.6506"
