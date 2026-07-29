#!/usr/bin/env bash
# Unattended pod driver: provision -> smoke-test -> SSS production run, as ONE
# detached chain, so the GPU never waits on a human between stages.
#
#   usage (on the pod, detached):
#     cd ~/sss_prod && nohup ./run_unattended.sh chair > unattended.log 2>&1 &
#
# Why this exists (each clause paid for by a real wasted-money incident):
#
#  * The bonsai run launched run_sss_production.sh before provisioning had
#    finished, so train.py died with `ModuleNotFoundError: diff_t_rasterization`
#    THREE times -- burning attempts, not iterations, on a billing GPU. Here the
#    two are chained in one process and gated by an explicit import smoke test
#    that aborts loudly instead of falling through into the retry loop.
#
#  * RENT_GUIDE gotcha #4: a monitor that greps only for a completion line misses
#    a stage that starts and then dies. So every stage publishes a MARKER FILE
#    (.done_ply / .done_renders / .done_all / .FAILED) plus a one-line HEARTBEAT
#    the watcher can poll for pennies. .done_ply lands the moment the ply is
#    written -- the watcher pulls the 1 GB ply while stage 2 is still rendering.
#
#  * A hung stage would otherwise bill until someone notices. MAX_HOURS caps it:
#    on expiry the chain writes .FAILED so the watcher escalates immediately.
set -uo pipefail

SCENE=${1:?usage: run_unattended.sh <scene>}
ROOT=${ROOT:-$HOME/sss_prod}
OUT=$ROOT/out/$SCENE
MAX_HOURS=${MAX_HOURS:-8}
HB=$OUT/HEARTBEAT
mkdir -p "$OUT"
rm -f "$OUT/.FAILED" "$OUT/.done_ply" "$OUT/.done_renders" "$OUT/.done_all"

START=$(date +%s)
say() { echo "[$(date -u +%H:%M:%SZ)] $*"; printf '%s | %s\n' "$(date -u +%H:%M:%SZ)" "$*" > "$HB"; }
die() { say "FATAL: $*"; echo "$*" > "$OUT/.FAILED"; exit 1; }

# Kill the whole chain if it outlives its budget, so a hang cannot bill overnight.
( sleep $((MAX_HOURS * 3600))
  [ -f "$OUT/.done_all" ] || { echo "exceeded MAX_HOURS=$MAX_HOURS" > "$OUT/.FAILED"; pkill -P $$; }
) & WATCHDOG=$!
trap 'kill $WATCHDOG 2>/dev/null || true' EXIT

say "stage 0/3: provisioning conda env sss (CPU work; GPU idle ~30 min, unavoidable)"
ROOT=$ROOT bash "$ROOT/provision_pod_sss.sh" || die "provisioning failed"

say "stage 1/3: smoke test (the gate the bonsai run lacked)"
# Run the smoke test in a SUBSHELL so the activation does not leak into stage 2.
# run_sss_production.sh does its own activation, and an inherited CONDA_SHLVL
# turns that into a no-op while its PATH export promotes the BASE interpreter --
# which fails the dep check with a misleading "No module named 'cv2'".
( export PATH=/opt/miniconda/bin:$PATH
  source /opt/miniconda/etc/profile.d/conda.sh
  conda activate sss || exit 1
  python - <<'PY'
import torch, cv2, diff_t_rasterization, simple_knn  # noqa: F401
assert torch.cuda.is_available(), "no CUDA device visible"
print(f"   torch {torch.__version__} | {torch.cuda.get_device_name(0)} | extensions OK")
PY
) || die "sss env incomplete -- refusing to start a billed train that would just retry-loop"

say "stage 2/3: SSS train + render (iters=${ITERS:-60000} cap=${CAP:-4000000})"
bash "$ROOT/run_sss_production.sh" "$SCENE" || die "production run failed -- see $OUT/train.log"

[ -f "$OUT/.done_all" ] || die "production run exited 0 but left no .done_all marker"
say "stage 3/3: COMPLETE in $(( ($(date +%s) - START) / 60 )) min -- deliverables ready, TERMINATE THE POD"
