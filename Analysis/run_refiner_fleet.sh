#!/usr/bin/env bash
# P2 fleet: train + apply the neural refiner on the remaining scenes (hcm0034 &
# HCM0181 already done), collect each scene's val_loss + (public) test Score,
# then build the refined submission. ~25-30 min/scene on the 1660 Ti.
set -u
cd /home/phong/Viettel_AI_Race_2026/bts-nvs
LOGDIR=Analysis/X5_refiner/fleet_logs
mkdir -p "$LOGDIR"
SUMMARY=Analysis/X5_refiner/fleet_summary.txt
: > "$SUMMARY"

# 3 remaining public (extra test-transfer datapoints) then 8 private
SCENES="hcm0031 HCM0193 HCM0204 HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437"

for s in $SCENES; do
  echo "===== refiner fleet: $s ($(date +%H:%M)) ====="
  conda run -n airace python Analysis/10_refiner_pilot.py --scene "$s" --iters 3000 \
      > "$LOGDIR/$s.log" 2>&1
  vloss=$(grep -oE "best val_loss [0-9.]+" "$LOGDIR/$s.log" | tail -1)
  score=$(grep -oE "Score=[0-9.]+" "$LOGDIR/$s.log" | tail -1)
  echo "$s: $vloss  $score" | tee -a "$SUMMARY"
done

echo "===== building refined submission ====="
conda run -n airace python Analysis/11_build_refined_submission.py 2>&1 | \
    grep -viE "FutureWarning|weights_only|torch.load|amp|malicious|allowlist|SECURITY|recommend|Arbitrary|flipped|default value|custom_" | tail -30
echo "FLEET_DONE"
