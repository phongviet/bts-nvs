#!/usr/bin/env bash
# exp034 v2 fleet: sharper inputs (ss=2 supersampled canvas + cubic gather) +
# refiner v2 (base 48, 6k iters, EMA, deterministic val) + TTA apply, per scene.
# Default scene list = the 8 private (the LB movers); pass scenes as args to
# override (e.g. public scenes for validation).
# ~2h/scene on the 1660 Ti -> private-8 ~16h. On a rented 4090 ~25 min/scene.
# Usage: bash Analysis/run_v2_fleet.sh [scene ...]
set -u
cd "$(dirname "$0")/.."
LOGDIR=Analysis/X5_refiner/v2_fleet_logs
mkdir -p "$LOGDIR"
SUMMARY=Analysis/X5_refiner/v2_fleet_summary.txt
touch "$SUMMARY"

SCENES="${@:-HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437}"

# Optional per-scene backbone override (exp034_private_big_fleet checkpoints).
# When runs/phase1/exp034_private_big_fleet/<scene>/big_30k exists it is used;
# otherwise the exp005 antialiased checkpoint from CONFIGS applies.
bb_flag() {
  local d="runs/phase1/exp034_private_big_fleet/$1/big_30k"
  [ -d "$d" ] && echo "--config $d" || echo ""
}

for s in $SCENES; do
  echo "===== v2 fleet: $s ($(date +%H:%M)) ====="
  # shellcheck disable=SC2046
  conda run -n airace python Analysis/10_refiner_pilot.py --scene "$s" \
      --iters 6000 --base 48 --ema 0.999 --tta \
      --ss 2 --sample cubic $(bb_flag "$s") --suffix _v2 \
      > "$LOGDIR/$s.log" 2>&1
  vloss=$(grep -oE "best val_loss [0-9.]+" "$LOGDIR/$s.log" | tail -1)
  score=$(grep -oE "Score=[0-9.]+" "$LOGDIR/$s.log" | tail -1)
  echo "$s: $vloss  $score  ($(date +%H:%M))" | tee -a "$SUMMARY"
done

echo "===== v2 fleet done -> build with 14_build_v2_submission.py ====="
