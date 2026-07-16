#!/usr/bin/env bash
# Local wave 2026-07-17: everything runnable on the 6GB card, serialized.
#   1. round-2 chair end-to-end (portrait pipeline validation)
#   2. exp037 encode knapsack self-test + real-GT A/B          (Analysis/16+19)
#   3. exp036 DIBR K=5 + IBGS rel-tol sweep, pilot hcm0034     (Analysis/04)
#   4. exp039 flow-residual alignment pilot (DIS backend)      (Analysis/04+17)
#   5. exp036 refiner seed-2 member (v2 recipe) for R2 ensemble (Analysis/10)
#   6. round-2 bonsai end-to-end
#   7. exp040 refiner v3 pilot (NAFNet blocks + evidence stack) (Analysis/10)
# Every step is independent: a failure logs and the queue continues.
# Logs: results/local_wave_jul17/<step>.log
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

LOG=results/local_wave_jul17
mkdir -p "$LOG"

step() {  # step <name> <cmd...>
  local name=$1; shift
  if [ -f "$LOG/$name.done" ]; then echo "== SKIP $name (done)"; return 0; fi
  echo "== START $name: $* [$(date +%H:%M)]"
  if "$@" > "$LOG/$name.log" 2>&1; then
    touch "$LOG/$name.done"; echo "== DONE $name [$(date +%H:%M)]"
  else
    echo "== FAIL $name (exit $?) — see $LOG/$name.log [$(date +%H:%M)]"
  fi
}

# --- 1. round-2 chair (portrait; SIMPLE_PINHOLE -> no remap downstream) ---
step 10_chair scripts/phase_run.sh chair all round2

# --- 2. exp037: equal-bytes real-GT A/B (drives 16_encode_knapsack internally) ---
step 21_encode_ab python Analysis/19_encode_ab.py --scene hcm0034

# --- 3. exp036: DIBR K=5 + relative-tau depth-consistency filter ---
step 30_dibr_base python Analysis/04_x3_dibr_pilot.py --scene hcm0034 --mode traincheck --K 3 --tol 0.03
step 31_dibr_k5_tau5e-4 python Analysis/04_x3_dibr_pilot.py --scene hcm0034 --mode traincheck --K 5 --rel-tol 5e-4
step 32_dibr_k5_tau1e-3 python Analysis/04_x3_dibr_pilot.py --scene hcm0034 --mode traincheck --K 5 --rel-tol 1e-3
step 33_dibr_k5_tau2e-3 python Analysis/04_x3_dibr_pilot.py --scene hcm0034 --mode traincheck --K 5 --rel-tol 2e-3
step 34_dibr_k5_test python Analysis/04_x3_dibr_pilot.py --scene hcm0034 --mode test --K 5 --rel-tol 1e-3 --vtag _k5rel1e3

# --- 4. exp039: flow alignment (DIS = compliant, runnable today) ---
step 40_flow_traincheck python Analysis/04_x3_dibr_pilot.py --scene hcm0034 --mode traincheck --K 5 --rel-tol 1e-3 --flow-align dis
step 41_flow_test python Analysis/04_x3_dibr_pilot.py --scene hcm0034 --mode test --K 5 --rel-tol 1e-3 --flow-align dis --vtag _k5rel1e3_dis

# --- 5. exp036 R2: second refiner seed (v2 recipe), ensemble scored separately ---
step 50_refiner_seed1 python Analysis/10_refiner_pilot.py --scene hcm0034 \
  --iters 6000 --base 48 --bs 4 --ema 0.999 --tta --seed 1 --suffix _v2s1
step 51_seed_ensemble python Analysis/20_seed_ensemble.py --scene hcm0034 \
  --members renders_refined_v2 renders_refined_v2s1 --out renders_refined_v2ens

# --- 6. round-2 bonsai ---
step 60_bonsai scripts/phase_run.sh bonsai all round2

# --- 7. exp040: refiner v3 (NAFNet + evidence stack on aligned warps) ---
step 70_refiner_v3 python Analysis/10_refiner_pilot.py --scene hcm0034 \
  --iters 6000 --base 48 --bs 2 --ema 0.999 --tta --blocks naf --evidence \
  --rel-tol 1e-3 --flow-align dis --max-pairs 60 --suffix _v3naf

echo "== QUEUE COMPLETE [$(date)] =="
