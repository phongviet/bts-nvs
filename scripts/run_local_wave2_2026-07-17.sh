#!/usr/bin/env bash
# Local wave 2 (follow-up, Jul-17): fixes/compositions from wave-1 readouts.
#   - chair retry (phase_run set-e bug fixed; resumes at training)
#   - exp036: tighter tau probe + GUARDED test compositions (wave-1 tests ran
#     guard-off: rel-tol alone tied the guarded baseline; flow guard-off LOST)
#   - exp039: flow with guard engaged = the actual GADA mechanism test
# Logs: results/local_wave_jul17/<step>.log (same dir, 8x-prefixed steps)
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

LOG=results/local_wave_jul17
mkdir -p "$LOG"

step() {
  local name=$1; shift
  if [ -f "$LOG/$name.done" ]; then echo "== SKIP $name (done)"; return 0; fi
  echo "== START $name: $* [$(date +%H:%M)]"
  if "$@" > "$LOG/$name.log" 2>&1; then
    touch "$LOG/$name.done"; echo "== DONE $name [$(date +%H:%M)]"
  else
    echo "== FAIL $name (exit $?) — see $LOG/$name.log [$(date +%H:%M)]"
  fi
}

# --- chair retry (dense init cached; resumes at ns-train) ---
step 80_chair_retry scripts/phase_run.sh chair all round2

# --- exp036: is the tau optimum tighter than 5e-4? ---
step 81_dibr_k5_tau2p5e-4 python Analysis/04_x3_dibr_pilot.py --scene hcm0034 --mode traincheck --K 5 --rel-tol 2.5e-4

# --- exp036: guarded compositions at test level (baseline = 0.7410 g0.18) ---
step 82_test_g_k5_tau5e-4 python Analysis/04_x3_dibr_pilot.py --scene hcm0034 --mode test --guard 0.18 --K 5 --rel-tol 5e-4 --vtag _g18k5rel5e4
step 83_test_g_k5_tau1e-3 python Analysis/04_x3_dibr_pilot.py --scene hcm0034 --mode test --guard 0.18 --K 5 --rel-tol 1e-3 --vtag _g18k5rel1e3

# --- exp039: flow WITH guard (the mechanism: rescue guard-rejected pixels) ---
step 84_tc_g_flow python Analysis/04_x3_dibr_pilot.py --scene hcm0034 --mode traincheck --guard 0.18 --K 5 --rel-tol 5e-4 --flow-align dis
step 85_test_g_flow python Analysis/04_x3_dibr_pilot.py --scene hcm0034 --mode test --guard 0.18 --K 5 --rel-tol 5e-4 --flow-align dis --vtag _g18k5rel5e4_dis

echo "== WAVE-2 QUEUE COMPLETE [$(date)] =="
