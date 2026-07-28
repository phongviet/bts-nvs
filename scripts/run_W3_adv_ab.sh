#!/usr/bin/env bash
# W3 — adversarial refiner gate.
#
# The refiner's loss (0.4*LPIPS + 0.3*(1-SSIM) + 0.3*L1) is pure regression, so it
# converges to the conditional MEAN and is systematically soft -- which is what an
# indoor LPIPS of 0.28-0.30 measures. A critic supplies the missing "is this a
# plausible photo" gradient. The grader's weights make the trade favourable:
# -1.0 dB PSNR costs 0.006 Score, -0.05 LPIPS gains 0.020.
#
# Arms are built ON TOP OF the W1-adopted naf+evidence config (chair 0.6696 /
# bonsai 0.6952), warm-started from that arm's weights -- a critic on a randomly
# initialised net just trains the critic.
#
#   w3_ctrl : naf+evidence, no critic  (== w1_naf; re-run so the panel is self-contained)
#   w3_a003 : --adv 0.003   (conservative)
#   w3_a010 : --adv 0.01    (the nominal bet)
#
# ADOPT: >= +0.002 Score vs w3_ctrl AND no LPIPS regression.
# usage: run_W3_adv_ab.sh <scene>
set -uo pipefail
cd "$(dirname "$0")/.."
for CB in /opt/miniconda "$HOME/miniconda3" "$(conda info --base 2>/dev/null)"; do
  [ -n "$CB" ] && [ -f "$CB/etc/profile.d/conda.sh" ] && { export PATH="$CB/bin:$PATH"; . "$CB/etc/profile.d/conda.sh"; break; }
done
conda activate airace || { echo "FATAL: cannot activate airace"; exit 1; }
python -c 'import nerfstudio' 2>/dev/null || { echo "FATAL: nerfstudio missing"; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SCENE="${1:?usage: run_W3_adv_ab.sh <scene>}"
case "$SCENE" in
  chair)  BB=runs/round2/val_holdout/chair/train_staging_holdout/splatfacto/2026-07-19_094122 ;;
  bonsai) BB=runs/round2/val_holdout/bonsai/train_staging_holdout/splatfacto/2026-07-18_003922 ;;
  *) echo "FATAL: unknown scene $SCENE"; exit 1 ;;
esac
[ -d "$BB" ] || { echo "FATAL: hold-out backbone missing: $BB"; exit 1; }

WARM="Analysis/X5_refiner/$SCENE/refiner_w1_naf.pt"
[ -f "$WARM" ] || { echo "FATAL: warm-start weights missing: $WARM (run W1 first)"; exit 1; }

LOG="results/W3_${SCENE}_ab"; mkdir -p "$LOG"

# identical to the W1-adopted arm in every respect except the critic
COMMON="--scene $SCENE --val-holdout --config $BB --ss 2 --sample cubic \
        --base 48 --bs 2 --iters 6000 --ema 0.999 --tta --max-pairs 90 \
        --blocks naf --evidence"

# SERIAL: arms share X5_refiner/<scene>/depth_cache<tag>; running them together
# corrupts the .npy cache (this cost a full W1 round).
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

run w3_ctrl
run w3_a003 --adv 0.003 --init-from "$WARM"
run w3_a010 --adv 0.01  --init-from "$WARM"

echo
echo "================ W3 VERDICT ($SCENE, 25 hold-out) ================"
python - "$SCENE" <<'PY'
import json, os, sys
scene = sys.argv[1]
OUT = f"Analysis/X5_refiner/{scene}"
arms = ["w3_ctrl", "w3_a003", "w3_a010"]
rows = {}
for a in arms:
    p = f"{OUT}/metrics_val_{a}.json"
    if os.path.exists(p):
        d = json.load(open(p)); rows[a] = d.get("mean", d)
if "w3_ctrl" not in rows:
    print("  control missing"); raise SystemExit
c = rows["w3_ctrl"]
print(f"  {'arm':10s} {'Score':>8s} {'dScore':>8s} {'PSNR':>7s} {'LPIPS':>8s} {'dLPIPS':>8s}   verdict")
for a in arms:
    m = rows.get(a)
    if m is None:
        print(f"  {a:10s}   (missing)"); continue
    ds, dl = m["score"] - c["score"], m["lpips"] - c["lpips"]
    v = "" if a == "w3_ctrl" else ("ADOPT" if ds >= 0.002 and dl <= 0 else "reject")
    print(f"  {a:10s} {m['score']:8.4f} {ds:+8.4f} {m['psnr']:7.3f} {m['lpips']:8.4f} {dl:+8.4f}   {v}")
PY
echo "=================================================================="
