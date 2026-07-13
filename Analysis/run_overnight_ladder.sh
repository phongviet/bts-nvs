#!/usr/bin/env bash
# Overnight ladder orchestrator (Jul-12): serialize the remaining pilot
# measurements after the in-flight E3 arms + E5a finish. RAM (11 GB) allows
# only one heavy job at a time.
#   1. wait for E3 (4 DIBR arms, hcm0034) + E5a (v2 refiner on v1 inputs)
#   2. E4b: HCM0181 DIBR ss2+cubic on antialiased AND on big_100k
#   3. E5b: hcm0034 refiner v2 on ss2+cubic inputs (+big backbone variant)
#   4. E5c: HCM0181 refiner v2 on ss2+cubic+big_100k inputs
# Each step logs to Analysis/X5_refiner/overnight/ and appends a one-liner to
# overnight_summary.txt.
set -u
cd "$(dirname "$0")/.."
OD=Analysis/X5_refiner/overnight
mkdir -p "$OD"
SUM=$OD/overnight_summary.txt
touch "$SUM"
note() { echo "[$(date +%H:%M)] $*" | tee -a "$SUM"; }

# --- 1. wait for in-flight jobs (E3 arms write 4 'DIBR:' lines; E5a writes a
#        REFINER line or the process dies) ---
note "waiting for E3 (4 arms) + E5a ..."
while :; do
  n_dibr=$(grep -c "DIBR: PSNR" Analysis/X3_dibr/e3_arms.log 2>/dev/null || true)
  e5a_done=0
  pgrep -f "suffix _v2" >/dev/null 2>&1 || e5a_done=1
  [ "$n_dibr" -ge 4 ] && [ "$e5a_done" -eq 1 ] && break
  sleep 60
done
note "E3+E5a done (E3 arms: $n_dibr)"

# --- 2. E4b: HCM0181 DIBR arms ---
note "E4b: HCM0181 ss2+cubic (antialiased) ..."
conda run -n airace python Analysis/04_x3_dibr_pilot.py --scene HCM0181 --mode test \
    --guard 0.18 --ss 2 --sample cubic --vtag _ss2cub > "$OD/hcm0181_ss2cub.log" 2>&1
grep -E "DIBR: PSNR" "$OD/hcm0181_ss2cub.log" | tail -1 | tee -a "$SUM"

note "E4b: HCM0181 ss2+cubic (big_100k) ..."
conda run -n airace python Analysis/04_x3_dibr_pilot.py --scene HCM0181 --mode test \
    --guard 0.18 --ss 2 --sample cubic \
    --config runs/phase1/exp006_capacity_iters_sweep/HCM0181/big_100k \
    --vtag _ss2cubbig > "$OD/hcm0181_ss2cubbig.log" 2>&1
grep -E "DIBR: PSNR" "$OD/hcm0181_ss2cubbig.log" | tail -1 | tee -a "$SUM"

# --- 3. E5b: hcm0034 refiner v2 on ss2+cubic inputs ---
note "E5b: hcm0034 refiner v2 on ss2+cubic ..."
conda run -n airace python Analysis/10_refiner_pilot.py --scene hcm0034 \
    --iters 6000 --base 48 --ema 0.999 --tta --ss 2 --sample cubic \
    --suffix _v2sc > "$OD/hcm0034_v2sc.log" 2>&1
grep -E "best val_loss|REFINER" "$OD/hcm0034_v2sc.log" | tail -2 | tee -a "$SUM"

# --- 3b. hcm0034 with big_60k backbone on top ---
note "E5b+bb: hcm0034 refiner v2 on ss2+cubic+big_60k ..."
conda run -n airace python Analysis/10_refiner_pilot.py --scene hcm0034 \
    --iters 6000 --base 48 --ema 0.999 --tta --ss 2 --sample cubic \
    --config runs/phase1/exp006_capacity_iters_sweep/hcm0034/big_60k \
    --suffix _v2scb > "$OD/hcm0034_v2scb.log" 2>&1
grep -E "best val_loss|REFINER" "$OD/hcm0034_v2scb.log" | tail -2 | tee -a "$SUM"

# --- 4. E5c: HCM0181 refiner v2 full stack ---
note "E5c: HCM0181 refiner v2 on ss2+cubic+big_100k ..."
conda run -n airace python Analysis/10_refiner_pilot.py --scene HCM0181 \
    --iters 6000 --base 48 --ema 0.999 --tta --ss 2 --sample cubic \
    --config runs/phase1/exp006_capacity_iters_sweep/HCM0181/big_100k \
    --suffix _v2scb > "$OD/hcm0181_v2scb.log" 2>&1
grep -E "best val_loss|REFINER" "$OD/hcm0181_v2scb.log" | tail -2 | tee -a "$SUM"

note "OVERNIGHT LADDER COMPLETE"
