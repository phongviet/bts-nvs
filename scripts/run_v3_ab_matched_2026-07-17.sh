#!/usr/bin/env bash
# exp040 refiner v3 A/B, matched-pair-count (2026-07-17).
#
# WHY 60 PAIRS: the v3 evidence stack is 20ch vs v2's 7ch. `sample_batch` holds
# the whole pair pool in RAM, so 240 pairs costs 6.3 GB at 7ch but 14.4 GB at
# 20ch -- and this box has 11 GB total. Full-pair v3 is OOM-killed by the kernel
# every time (confirmed: dmesg "Out of memory: Killed process ... anon-rss:9.5GB").
# So 60 pairs is a HARD CONSTRAINT for v3, not a shortcut. The published v2
# number (0.7710) used all 240 pairs -> comparing v3@60 against it confounds
# architecture with training-set size. This script re-measures v2 at 60 pairs so
# the A/B varies ONLY --blocks/--evidence.
#
# Both arms sit on the plain guarded base (guard=0.18 default, no --rel-tol, no
# --flow-align) because exp039 refuted flow alignment on this scene.
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

LOG=results/local_wave_jul17
COMMON="--scene hcm0034 --iters 6000 --base 48 --bs 2 --ema 0.999 --tta --max-pairs 60"

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

# control: v2 recipe (7ch, conv blocks) at the SAME 60 pairs as v3
run 91_v2_ctrl_p60 --blocks conv --suffix _v2ctrl_p60

# arm: v3 (20ch evidence stack, naf blocks)
run 92_v3_naf_ev_p60 --blocks naf --evidence --suffix _v3naf_p60

echo "== A/B COMPLETE [$(date)] =="
