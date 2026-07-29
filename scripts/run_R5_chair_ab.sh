#!/usr/bin/env bash
# R5 (reserve ladder §4) -- chair sharpness-stratified refiner pairs A/B.
#
# WHY: submission R2-2 (indoor max-pairs 90->180) moved chair not at all
# (val_loss 0.1230->0.1231) and the LB gain came entirely from SSIM/PSNR, never
# LPIPS -- pair COUNT is exhausted as an indoor lever. Chair's deficit is MOTION
# BLUR (train varLap 1502 early -> 344 late), so the remaining bet is pair
# SELECTION: train the refiner on sharper supervision and see whether it learns
# a render->crisp map that lowers LPIPS on the held-out (mildly-blurred) frames.
#
# DESIGN: fixed 25-frame chair holdout, train-minus-holdout backbone, shipped
# resampling (ss2+cubic), everything identical across arms EXCEPT --pair-select
# and --max-pairs. 90 of the 180 eligible views so the selection actually bites.
#   stride180 = the shipped reference (all views, capture-order)
#   stride090 = count control (does halving the count alone hurt? R2-2 says no)
#   sharp090  = the 90 sharpest views
#   strat090  = 90 stratified across the sharpness spectrum
#
# ADOPT (per §5, +0.002 Score bar) only if max(sharp,strat)@90 beats BOTH
# stride180 (no regression vs shipped) AND stride090 (effect is selection, not
# count). Otherwise chair's LPIPS floor is scene-intrinsic -> R5 dead, freeze.
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

LOG=results/R5_chair_ab
mkdir -p "$LOG"

BB=runs/round2/val_holdout/chair/train_staging_holdout/splatfacto/2026-07-19_094122
[ -d "$BB" ] || { echo "FATAL: hold-out backbone missing: $BB"; exit 1; }

# ss2+cubic = shipped; bs=2 for the 720x1280 portrait on the 6 GB card.
COMMON="--scene chair --val-holdout --config $BB --ss 2 --sample cubic \
        --base 48 --bs 2 --iters 6000 --ema 0.999 --tta --blocks conv"

run() {  # run <name> <max_pairs> <select>
  local name=$1 mp=$2 sel=$3
  if [ -f "$LOG/$name.done" ]; then echo "== SKIP $name (done)"; return 0; fi
  echo "== START $name (max-pairs $mp, $sel) [$(date +%H:%M)]"
  # shellcheck disable=SC2086
  if python Analysis/10_refiner_pilot.py $COMMON \
        --max-pairs "$mp" --pair-select "$sel" --suffix "_$name" \
        > "$LOG/$name.log" 2>&1; then
    touch "$LOG/$name.done"; echo "== DONE $name [$(date +%H:%M)]"
  else
    echo "== FAIL $name (exit $?) -- see $LOG/$name.log [$(date +%H:%M)]"
  fi
}

# stride180 FIRST: max-pairs==available -> keep=None -> caches ALL 180 pairs,
# which the 90-view arms then reuse for free (shared cache, identical content).
run r5_stride180 180 stride
run r5_stride090  90 stride
run r5_sharp090   90 sharp
run r5_strat090   90 strat

echo
echo "== R5 CHAIR A/B SUMMARY (holdout Score, higher=better; +0.002 = adopt) =="
for a in r5_stride180 r5_stride090 r5_sharp090 r5_strat090; do
  f="Analysis/X5_refiner/chair/metrics_val_$a.json"
  if [ -f "$f" ]; then
    python - "$a" "$f" <<'PY'
import json,sys
a,f=sys.argv[1],sys.argv[2]; m=json.load(open(f))
print(f"  {a:14s} Score {m['score']:.4f}  PSNR {m['psnr']:.3f}  SSIM {m['ssim']:.4f}  LPIPS {m['lpips']:.4f}")
PY
  else
    echo "  $a  (no metrics -- run failed?)"
  fi
done
echo "Backbone-only reference on the same 25 frames: Score 0.6506."
