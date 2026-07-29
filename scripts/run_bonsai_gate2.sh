#!/usr/bin/env bash
# GATE-2: does the SSS backbone survive the refiner? Full-stack bonsai holdout A/B.
#
# Two refiner runs, IDENTICAL config (the shipped bonsai holdout config), differing
# ONLY in the render channel of the 7-ch input:
#   g2ctrl = splatfacto-big rgb_T (the 3DGS backbone)  -> reproduces the control 0.6913
#   g2sss  = SSS renders via --render-override          -> the treatment
# depth_T stays from the splatfacto holdout backbone in BOTH arms (DIBR geometry
# unchanged); only the render channel + DIBR fallback/exposure ref changes.
#
# ADOPT (proceed to fleet-integrate + submission) only if g2sss beats g2ctrl full-stack
# by >= +0.002 Score AND does not regress LPIPS. Else the refiner ABSORBS the backbone
# gain -> bonsai backbone swap does not ship; freeze R2-2.
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate airace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

LOG=results/bonsai_gate2; mkdir -p "$LOG"
BB=runs/round2/val_holdout/bonsai/train_staging_holdout/splatfacto/2026-07-18_003922
[ -d "$BB" ] || { echo "FATAL: holdout backbone missing: $BB"; exit 1; }
RO=Analysis/X5_refiner/bonsai/sss_renders_ho
[ "$(ls "$RO" 2>/dev/null | wc -l)" -ge 248 ] || { echo "FATAL: SSS renders incomplete in $RO"; exit 1; }

# Exact control config (scripts/run_refiner_phase1_bonsai_2026-07-19.sh): ss=1, bilinear.
COMMON="--scene bonsai --val-holdout --config $BB --iters 6000 --base 48 --bs 2 \
        --ema 0.999 --tta --max-pairs 100"

run() {  # run <suffix> [extra args...]
  local name=$1; shift
  if [ -f "$LOG/$name.done" ]; then echo "== SKIP $name"; return 0; fi
  echo "== START $name [$(date +%H:%M)] =="
  # shellcheck disable=SC2086
  if python Analysis/10_refiner_pilot.py $COMMON --suffix "_$name" "$@" > "$LOG/$name.log" 2>&1; then
    touch "$LOG/$name.done"; echo "== DONE $name [$(date +%H:%M)] =="
  else
    echo "== FAIL $name (exit $?) -- see $LOG/$name.log"; tail -5 "$LOG/$name.log"
  fi
}

run g2ctrl
run g2sss --render-override "$RO"

echo
echo "================ GATE-2 VERDICT (bonsai, 25 holdout, full-stack) ================"
python - <<'PY'
import json, os
OUT="Analysis/X5_refiner/bonsai"
def load(s):
    p=f"{OUT}/metrics_val_{s}.json"
    if not os.path.exists(p): return None
    d=json.load(open(p)); return d.get("mean", d)
c, t = load("g2ctrl"), load("g2sss")
if not c or not t:
    print("  missing metrics -- a run failed; see results/bonsai_gate2/*.log"); raise SystemExit
def row(n,m): print(f"  {n:10s} Score {m['score']:.4f}  PSNR {m['psnr']:.3f}  SSIM {m['ssim']:.4f}  LPIPS {m['lpips']:.4f}")
row("splat(ctrl)", c); row("SSS(g2)", t)
ds=t['score']-c['score']; dl=t['lpips']-c['lpips']
print(f"  ΔScore {ds:+.4f}   ΔLPIPS {dl:+.4f} (neg = SSS better)")
print(f"  reference: splat backbone-only 0.6759, SSS backbone-only 0.6993, shipped ctrl 0.6913")
ok = ds >= 0.002 and dl <= 0.0
print(f"  GATE (+0.002 Score AND no LPIPS regression): "
      f"{'PASS -> fleet-integrate SSS bonsai + submission' if ok else 'FAIL -> refiner absorbs it, freeze R2-2'}")
PY
echo "================================================================================"
