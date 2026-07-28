#!/usr/bin/env bash
# Refiner Phase 1 (2026-07-19): put the indoor regime on the recipe panel.
#
# Every refiner decision so far -- v2's recipe, exp036's guard+K5+rel-tau, exp040's
# v3 evidence stack -- was measured on hcm0034 alone: one quiet drone scene. Two
# of the seven graded round-2 scenes are handheld indoor video, and the bonsai
# hold-out (Analysis/VAL_bonsai_holdout_2026-07-18.md) says that regime behaves
# differently: backbone LPIPS 0.323 vs ~0.24 on drone scenes, and 32% of test
# poses sit in a region with ~2 nearby train views instead of 21.
#
# That last fact is the one this run exists to test. DIBR and the refiner both
# consume NEIGHBOURING REAL VIEWS, so where neighbours are missing they have
# nothing to work with -- the opposite of the drone scenes, where DIBR gained
# most on the hardest frames. This measures whether the stack transfers at all.
#
# Honesty of the number rests on two exclusions, both wired in:
#   --config      -> the train-minus-holdout backbone (never saw the val frames)
#   --val-holdout -> Warper(holdout_names=...) so no val frame is a DIBR source,
#                    and the pairs are built without them
# Backbone-only reference on the same 25 frames: Score 0.6759
# (PSNR 25.619 / SSIM 0.8382 / LPIPS 0.3232).
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

LOG=results/local_wave_jul19
mkdir -p "$LOG"

BB=runs/round2/val_holdout/bonsai/train_staging_holdout/splatfacto/2026-07-18_003922
[ -d "$BB" ] || { echo "FATAL: hold-out backbone missing: $BB"; exit 1; }

# max-pairs is now a COMPUTE budget, not a RAM one: PairPool memmaps the cache,
# so pair count no longer drives residency. Pair BUILD still costs one warp per
# frame, and bonsai is 1920x1080, so start at 100 of the 223 available.
COMMON="--scene bonsai --val-holdout --config $BB --iters 6000 --base 48 --bs 2 --ema 0.999 --tta --max-pairs 100"

run() {
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

# v2 recipe, exactly as shipped on the drone scenes -- does it transfer?
run 11_bonsai_v2 --blocks conv --suffix _v2

echo "== PHASE1 BONSAI COMPLETE [$(date)] =="
echo "Compare against backbone-only 0.6759 on the same 25 frames."
