#!/usr/bin/env bash
# W9 — chair adversarial-weight ladder (the audit of "0.003 beats 0.01").
#
# Why this exists (Jul-24 audit): the recorded rule "--adv 0.003 BEATS --adv 0.01 on
# both scenes: do not push the weight higher" is UNSUPPORTED —
#   * chair's w3_a010 arm was the arm silently killed by the stale pod upload and
#     NEVER re-run (no metrics_val_w3_a010 exists anywhere in the repo);
#   * bonsai's a003-vs-a010 gap is −0.0013, inside the project's own ±0.002 bar,
#     on a hold-out that under-predicts shipped adversarial gains ~2x;
#   * drones shipped at 0.003 with no weight sweep at all, and their LB gain was
#     +0.0593, 100 % LPIPS — the exact trade the local rulers punish.
# Meanwhile v7a proved val-flavoured metrics are ANTI-correlated with LB for
# adversarial arms. So the weight was chosen with a broken ruler and the response
# curve above 0.003 is unknown. This ladder measures it, internally controlled:
# same machine, same bs, same warm ckpt, same pair cache, weight the only variable.
#
#   w9_ctrl : naf+evidence regression (stage 1) — reproduces w3_ctrl 0.6695
#             AND provides the warm ckpt for every arm
#   w9_a003 : --adv 0.003  (local anchor; pod result was 0.6724)
#   w9_a006 : --adv 0.006  (never tested anywhere)
#   w9_a010 : --adv 0.010  (the killed chair arm)
#
# READ-OUT RULE (v7a lesson): rank arms on the grader-shaped hold-out Score but
# treat LPIPS as the decision metric when Score is within ±0.002 — hold-out Score
# compresses adversarial gains ~2x and its PSNR term punishes exactly what LB pays
# for. A weight that ties on Score with better LPIPS is the LB-favourite.
# usage: run_W9_adv_ladder.sh   (chair only; ~8-10 h serial on the 1660 Ti)
set -uo pipefail
cd "$(dirname "$0")/.."
for CB in /opt/miniconda "$HOME/miniconda3" "$(conda info --base 2>/dev/null)"; do
  [ -n "$CB" ] && [ -f "$CB/etc/profile.d/conda.sh" ] && { export PATH="$CB/bin:$PATH"; . "$CB/etc/profile.d/conda.sh"; break; }
done
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate airace || { echo "FATAL: cannot activate airace"; exit 1; }
case "$(command -v python)" in
  "$CONDA_PREFIX"/bin/python) : ;;
  *) echo "FATAL: python is $(command -v python), not $CONDA_PREFIX/bin/python"; exit 1 ;;
esac
python -c 'import nerfstudio' 2>/dev/null || { echo "FATAL: nerfstudio missing"; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SCENE=chair
BB=runs/round2/val_holdout/chair/train_staging_holdout/splatfacto/2026-07-19_094122
[ -d "$BB" ] || { echo "FATAL: hold-out backbone missing: $BB"; exit 1; }

LOG="results/W9_${SCENE}_adv_ladder"; mkdir -p "$LOG"

# identical to the W3 arms in every respect (same flags as the shipped w5adv config)
COMMON="--scene $SCENE --val-holdout --config $BB --ss 2 --sample cubic \
        --base 48 --bs 2 --iters 6000 --ema 0.999 --tta --max-pairs 90 \
        --blocks naf --evidence"

# SERIAL: arms share the depth/pair caches; parallel arms corrupt the .npy files.
run() {
  local name=$1; shift
  if [ -f "$LOG/$name.done" ]; then echo "== SKIP $name"; return 0; fi
  echo "== START $name [$(date +%H:%M)] $*"
  # shellcheck disable=SC2086
  if python Analysis/10_refiner_pilot.py $COMMON --suffix "_$name" "$@" \
        > "$LOG/$name.log" 2>&1; then
    touch "$LOG/$name.done"; echo "== DONE $name [$(date +%H:%M)]"
  else
    echo "== FAIL $name (exit $?)"; tail -5 "$LOG/$name.log"
  fi
}

run w9_ctrl
WARM="Analysis/X5_refiner/$SCENE/refiner_w9_ctrl.pt"
[ -f "$WARM" ] || { echo "FATAL: stage-1 ckpt missing: $WARM"; exit 1; }
run w9_a003 --adv 0.003 --init-from "$WARM"
run w9_a006 --adv 0.006 --init-from "$WARM"
run w9_a010 --adv 0.01  --init-from "$WARM"

echo
echo "================ W9 VERDICT (chair, 25 hold-out) ================"
python - <<'PY'
import json, os
OUT = "Analysis/X5_refiner/chair"
arms = ["w9_ctrl", "w9_a003", "w9_a006", "w9_a010"]
rows = {}
for a in arms:
    p = f"{OUT}/metrics_val_{a}.json"
    if os.path.exists(p):
        d = json.load(open(p)); rows[a] = d.get("mean", d)
if "w9_ctrl" not in rows:
    print("  control missing"); raise SystemExit
c = rows["w9_ctrl"]
print("  pod refs: w3_ctrl 0.6695 / w3_a003 0.6724 (LPIPS 0.2747 -> 0.2706)")
print(f"  {'arm':10s} {'Score':>8s} {'dScore':>8s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>8s} {'dLPIPS':>8s}")
for a in arms:
    m = rows.get(a)
    if m is None:
        print(f"  {a:10s}   (missing)"); continue
    ds, dl = m["score"] - c["score"], m["lpips"] - c["lpips"]
    print(f"  {a:10s} {m['score']:8.4f} {ds:+8.4f} {m['psnr']:7.3f} {m['ssim']:7.4f} "
          f"{m['lpips']:8.4f} {dl:+8.4f}")
print("  read-out: within +-0.002 Score, the best-LPIPS weight is the LB-favourite")
PY
echo "=================================================================="
