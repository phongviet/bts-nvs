#!/usr/bin/env bash
# Gate A driver: bonsai then chair, serially (6 GB GPU fits one at a time).
# Each child script is idempotent via its own .done marker.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/local_wave_jul19
mkdir -p "$LOG"
echo "== GATE A START [$(date)] =="
bash scripts/run_refiner_phase1_bonsai_2026-07-19.sh 2>&1 | tee -a "$LOG/gateA_driver.log"
bash scripts/run_refiner_phase1_chair_2026-07-19.sh  2>&1 | tee -a "$LOG/gateA_driver.log"
echo "== GATE A COMPLETE [$(date)] =="
touch "$LOG/gateA.done"
