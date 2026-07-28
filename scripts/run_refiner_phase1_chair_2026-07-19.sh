#!/usr/bin/env bash
# Refiner Phase 1 -- chair (2026-07-19). Twin of run_refiner_phase1_bonsai.
#
# Chair is the #2 scene by score weight: 58 of 386 graded frames = 15.0% of the
# round-2 score, twice bonsai's 7.3%, and it is the WORST-scoring scene we have
# (hold-out Score 0.6506 vs bonsai 0.6759). So the indoor transfer question
# matters more here than on bonsai, and the failure mode is different.
#
# Bonsai fails from COVERAGE (32% of test poses have ~2 nearby train views).
# Chair does not: n_near is a uniform 3-4 everywhere. Chair fails from MOTION
# BLUR -- train-frame Laplacian variance decays along the capture (1502 early ->
# 663 mid -> 344 late, min 50), and val LPIPS tracks it (0.275 -> 0.333).
#
# That predicts the OPPOSITE sign of bonsai's risk. DIBR sources are temporally
# adjacent frames, so a blurry test pose is served by similarly blurry sources:
# the warp should be blur-MATCHED where the 3DGS render is blur-AVERAGED across
# a whole neighbourhood. If that reasoning holds, chair should gain MORE from
# the stack than bonsai, not less.
#
# Same two honesty exclusions as the bonsai run:
#   --config      -> train-minus-holdout backbone (never saw the val frames)
#   --val-holdout -> holdout frames excluded as DIBR sources AND from the pairs
# Backbone-only reference on the same 25 frames: Score 0.6506
# (PSNR 23.765 / SSIM 0.7688 / LPIPS 0.3065).
#
# NOTE: 720x1280 PORTRAIT. Any silent landscape assumption in the warper or the
# U-Net padding shows up here as a crash or a garbage score -- that is a second
# thing this run buys us before the fleet touches this scene.
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

LOG=results/local_wave_jul19
mkdir -p "$LOG"

BB=runs/round2/val_holdout/chair/train_staging_holdout/splatfacto/2026-07-19_094122
[ -d "$BB" ] || { echo "FATAL: hold-out backbone missing: $BB"; exit 1; }

# Chair is 0.92 MP/frame vs bonsai's 2.07 MP, so a pair costs ~2.2x less to
# build: budget 150 pairs of the ~180 available (bonsai runs 100).
COMMON="--scene chair --val-holdout --config $BB --iters 6000 --base 48 --bs 2 --ema 0.999 --tta --max-pairs 150"

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
run 12_chair_v2 --blocks conv --suffix _v2

echo "== PHASE1 CHAIR COMPLETE [$(date)] =="
echo "Compare against backbone-only 0.6506 on the same 25 frames."
