#!/usr/bin/env bash
# W1 — the four refiner levers that EXIST, ship OFF, and have only ever been
# A/B'd on drone scenes, where the physics that would make them work is absent.
#
#   w1_ctrl  : shipped config, the control (reproduces bonsai 0.6913 / chair 0.6661)
#   w1_flow  : --flow-align dis   -- refuted on drones because drone parallax ~ 0;
#              Analysis/25 proved indoor parallax is LARGE (a 2D temporal blend
#              scores 0.49/0.38 vs 0.69/0.67), so the premise is inverted here
#   w1_expo  : --exposure         -- handheld video auto-exposure/WB drifts between
#              frames; drone stills do not
#   w1_naf   : --blocks naf --evidence -- never run on indoor at all; naf is a
#              deblurring-family block and both indoor scenes are blur-limited
#
# Single-variable arms on the SAME fixed 25-frame match-test hold-out that every
# other indoor A/B used, so results drop straight onto the existing panel.
# ADOPT bar: >= +0.002 Score AND no LPIPS regression vs w1_ctrl.
#
# usage: run_W1_indoor_ab.sh <scene> [parallel]
set -uo pipefail
cd "$(dirname "$0")/.."
# conda is NOT on PATH in a non-login pod shell, so `conda info --base` fails and
# `conda activate` silently no-ops -> the job runs on base python and dies with
# ModuleNotFoundError: nerfstudio. Hardcode the pod prefix, fall back to local.
for CB in /opt/miniconda "$HOME/miniconda3" "$(conda info --base 2>/dev/null)"; do
  [ -n "$CB" ] && [ -f "$CB/etc/profile.d/conda.sh" ] && { export PATH="$CB/bin:$PATH"; . "$CB/etc/profile.d/conda.sh"; break; }
done
conda activate airace || { echo "FATAL: cannot activate airace"; exit 1; }
python -c 'import nerfstudio' 2>/dev/null || { echo "FATAL: nerfstudio missing in $(which python)"; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SCENE="${1:?usage: run_W1_indoor_ab.sh <scene> [parallel]}"
PAR="${2:-2}"

case "$SCENE" in
  chair)  BB=runs/round2/val_holdout/chair/train_staging_holdout/splatfacto/2026-07-19_094122 ;;
  bonsai) BB=runs/round2/val_holdout/bonsai/train_staging_holdout/splatfacto/2026-07-18_003922 ;;
  *) echo "FATAL: unknown scene $SCENE"; exit 1 ;;
esac
[ -d "$BB" ] || { echo "FATAL: hold-out backbone missing: $BB"; exit 1; }

LOG="results/W1_${SCENE}_ab"; mkdir -p "$LOG"

# ss2+cubic = shipped. max-pairs 90: R2-2 and R5 both showed count is saturated
# (90 -> 180 = +0.0003), so 90 is the cheap, equivalent operating point.
COMMON="--scene $SCENE --val-holdout --config $BB --ss 2 --sample cubic \
        --base 48 --bs 2 --iters 6000 --ema 0.999 --tta --max-pairs 90"

run() {  # run <name> [extra args...]
  local name=$1; shift
  if [ -f "$LOG/$name.done" ]; then echo "== SKIP $name (done)"; return 0; fi
  echo "== START $name [$(date +%H:%M)] $*"
  # shellcheck disable=SC2086
  if python Analysis/10_refiner_pilot.py $COMMON --suffix "_$name" "$@" \
        > "$LOG/$name.log" 2>&1; then
    touch "$LOG/$name.done"; echo "== DONE $name [$(date +%H:%M)]"
  else
    echo "== FAIL $name (exit $?) -- see $LOG/$name.log [$(date +%H:%M)]"
    tail -5 "$LOG/$name.log"
  fi
}

# ctrl first and ALONE: it builds the plain pair cache that w1_expo reuses, and
# racing four cache-building jobs on the same variant corrupts nothing but wastes
# the whole point of the shared cache.
run w1_ctrl

# SERIAL by default, and that is not conservatism: arms share
# X5_refiner/<scene>/depth_cache<tag>, and the tag only varies by warp VARIANT --
# so --flow-align / --exposure arms collide on the same .npy files and read each
# other's half-written arrays ("cannot reshape array of size N"). Only opt into
# parallelism once each arm has a private cache.
if [ "$PAR" -le 1 ]; then
  run w1_flow --flow-align dis
  run w1_expo --exposure
  run w1_naf  --blocks naf --evidence
else
  pids=()
  run w1_flow --flow-align dis &
  pids+=($!)
  run w1_expo --exposure &
  pids+=($!)
  for p in "${pids[@]}"; do wait "$p"; done
  run w1_naf --blocks naf --evidence
fi

echo
echo "================ W1 VERDICT ($SCENE, 25 hold-out) ================"
python - "$SCENE" <<'PY'
import json, os, sys
scene = sys.argv[1]
OUT = f"Analysis/X5_refiner/{scene}"
arms = ["w1_ctrl", "w1_flow", "w1_expo", "w1_naf"]
rows = {}
for a in arms:
    p = f"{OUT}/metrics_val_{a}.json"
    if os.path.exists(p):
        d = json.load(open(p)); rows[a] = d.get("mean", d)
if "w1_ctrl" not in rows:
    print("  control missing -- see results/W1_%s_ab/w1_ctrl.log" % scene); raise SystemExit
c = rows["w1_ctrl"]
print(f"  {'arm':10s} {'Score':>8s} {'dScore':>8s} {'LPIPS':>8s} {'dLPIPS':>8s}   verdict")
for a in arms:
    m = rows.get(a)
    if m is None:
        print(f"  {a:10s} {'--':>8s}   (missing)"); continue
    ds, dl = m["score"] - c["score"], m["lpips"] - c["lpips"]
    v = "" if a == "w1_ctrl" else ("ADOPT" if ds >= 0.002 and dl <= 0 else "reject")
    print(f"  {a:10s} {m['score']:8.4f} {ds:+8.4f} {m['lpips']:8.4f} {dl:+8.4f}   {v}")
PY
echo "=================================================================="
