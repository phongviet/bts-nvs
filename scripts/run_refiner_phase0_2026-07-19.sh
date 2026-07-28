#!/usr/bin/env bash
# Refiner Phase 0 (2026-07-19): prove the memmap loader, then run the experiment
# it unblocks.
#
# The old loader held every pair resident AND upcast f16->f32, so the pool cost
# 12.5 GB (v2, 7ch) / 28.8 GB (v3 evidence, 20ch) on an 11 GB box. That is why
# exp040 had to judge v3 at 60 pairs. PairPool memmaps per-pair .npy sidecars and
# slices only the 256px crop, making RAM O(1) in pair count.
#
# Arm A is a REGRESSION GATE, not a new datapoint: same recipe, same seed, same
# RNG draw order as the published v2 (0.7710). If it does not reproduce, the
# loader changed the numbers and every result below is void -- stop and debug.
#
# Arm B is the payoff: v3's evidence stack at the SAME 240 pairs as v2. exp040
# measured v3 at 1/4 the data of its control and still won by +0.0008; v2 itself
# gained +0.0015 going 60->240. If v3 scales at least as steeply, it clears the
# +0.002 bar. If it does not, the evidence stack is a dead end at any pair count.
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

LOG=results/local_wave_jul19
mkdir -p "$LOG"
COMMON="--scene hcm0034 --iters 6000 --base 48 --bs 2 --ema 0.999 --tta"

run() {  # run <name> <extra-args...>
  local name=$1; shift
  if [ -f "$LOG/$name.done" ]; then echo "== SKIP $name (done)"; return 0; fi
  echo "== START $name [$(date +%H:%M)]"
  # shellcheck disable=SC2086
  if python Analysis/10_refiner_pilot.py $COMMON "$@" > "$LOG/$name.log" 2>&1; then
    touch "$LOG/$name.done"; echo "== DONE $name [$(date +%H:%M)]"
  else
    echo "== FAIL $name (exit $?) -- see $LOG/$name.log [$(date +%H:%M)]"
  fi
}

# A: regression gate -- must land on 0.7710 (published v2 @ 240 pairs)
run 01_v2_mm_p240 --blocks conv --suffix _v2mm_p240

# B: the run the RAM ceiling forbade -- v3 evidence stack at full 240 pairs
run 02_v3_naf_ev_p240 --blocks naf --evidence --suffix _v3naf_p240

echo "== PHASE0 COMPLETE [$(date)] =="
