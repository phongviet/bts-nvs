#!/usr/bin/env bash
# GATE-1: SSS backbone-only vs splatfacto-big on the bonsai 25-frame holdout.
#
# Trains SSS on the bonsai hold-out (undistorted, train-minus-val), renders the
# 25 val poses, scores vs their real photos with the bts-nvs competition metric,
# and prints the verdict against the splatfacto-big control.
#
# CONTROL (already computed locally, runs/round2/val_holdout/bonsai/metrics_val_split.json):
#   Score 0.6759  PSNR 25.619  SSIM 0.8382  LPIPS 0.3232
# GATE: SSS must beat control Score by >= +0.003 (i.e. >= 0.6789) AND not regress
#   LPIPS (SSS habitually wins SSIM / loses LPIPS; LPIPS is the 0.4-weight metric
#   and the whole reason bonsai is the target) to justify paying for gate-2
#   (refiner integration). Anything less => SSS does not help bonsai => stop.
#
# Flags mirror the paper-scale fleet that produced the RESULTS.md "wash"
# (kaggle_sss_fleet.py): nu=100, C=120, C_burnin=5e5, burnin=7000, cap_max=2M, 40k.
set -euo pipefail

ROOT=${ROOT:-$HOME/sss_gate1}
SSS=$ROOT/3D-student-splatting-and-scooping
SRC=$ROOT/data/bonsai_ho_undist          # undistorted train-minus-val staging
POSES=$ROOT/data/bonsai_val_poses.csv     # 25 val poses (render_test_csv format)
GT=$ROOT/data/bonsai_val_gt               # 25 real val photos (matching filenames)
OUT=$ROOT/out/bonsai_sss_gate1
MODEL=$OUT/model
RENDERS=$OUT/renders_val
METRICS=$OUT/metrics_val_sss.json
LOG=$OUT/train.log
ITERS=${ITERS:-40000}
CAP=${CAP:-2000000}
mkdir -p "$OUT"

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate sss
export PYTHONPATH="$SSS:${PYTHONPATH:-}"
# per-iteration NaN guard is in-code; keep debug stats off for speed.

ply_glob() { ls "$MODEL"/point_cloud/iteration_*/point_cloud.ply 2>/dev/null | tail -1; }

# --- 1. train SSS (4-attempt auto-resume; SGHMC can still diverge rarely) ---
CKPTS=(); for i in $(seq 2500 2500 $((ITERS-1))); do CKPTS+=("$i"); done
if [ -z "$(ply_glob)" ]; then
  for attempt in 1 2 3 4; do
    [ -n "$(ply_glob)" ] && break
    echo "== SSS train attempt $attempt [$(date +%H:%M)] iters=$ITERS cap=$CAP =="
    RESUME=(); last=$(ls "$MODEL"/chkpnt*.pth 2>/dev/null | sort -t t -k3 -n | tail -1 || true)
    [ -n "$last" ] && { echo "   resuming from $last"; RESUME=(--start_checkpoint "$last"); }
    set +e
    python "$SSS/train.py" -s "$SRC" -m "$MODEL" --data_device cuda \
      --cap_max "$CAP" --nu_degree 100 --C_burnin 5e5 --C 1.2e2 \
      --burnin_iterations 7000 --iterations "$ITERS" \
      --save_iterations "$ITERS" --test_iterations "$ITERS" --quiet \
      --checkpoint_iterations "${CKPTS[@]}" "${RESUME[@]}" \
      >>"$LOG" 2>&1
    rc=$?; set -e
    echo "   attempt $attempt rc=$rc [$(date +%H:%M)]"
    [ "$attempt" = 4 ] && [ -z "$(ply_glob)" ] && { echo "FATAL: SSS train failed 4x -- see $LOG"; exit 1; }
  done
  rm -f "$MODEL"/chkpnt*.pth   # multi-GB, not needed once ply exists
fi
echo "== trained ply: $(ply_glob) =="

# --- 2. render the 25 val poses (nu-degree MUST match training = 100; q98 = control parity) ---
python "$ROOT/render_test_csv.py" --model "$MODEL" \
  --poses-csv "$POSES" --out "$RENDERS" --nu-degree 100
echo "== rendered $(ls "$RENDERS" | wc -l) val frames =="

# --- 3. score with the bts-nvs competition metric (vgg, psnr_max 50) ---
python "$ROOT/bts_metrics.py" --renders "$RENDERS" --gt "$GT" \
  --out "$METRICS" --lpips-net vgg --psnr-max 50.0

# --- 4. verdict ---
python - "$METRICS" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["mean"]
CTRL = dict(score=0.6758987, psnr=25.6188, ssim=0.8382, lpips=0.3231822)
ds = m["score"] - CTRL["score"]; dl = m["lpips"] - CTRL["lpips"]
print("\n================ GATE-1 VERDICT (bonsai, 25 holdout frames) ================")
print(f"  SSS      Score {m['score']:.4f}  PSNR {m['psnr']:.3f}  SSIM {m['ssim']:.4f}  LPIPS {m['lpips']:.4f}")
print(f"  splat-big Score {CTRL['score']:.4f}  PSNR {CTRL['psnr']:.3f}  SSIM {CTRL['ssim']:.4f}  LPIPS {CTRL['lpips']:.4f}")
print(f"  ΔScore {ds:+.4f}   ΔLPIPS {dl:+.4f} (negative = SSS better on LPIPS)")
passed = ds >= 0.003 and dl <= 0.0
print(f"  GATE (+0.003 Score AND no LPIPS regression): {'PASS -> proceed to gate-2' if passed else 'FAIL -> SSS does not help bonsai, stop'}")
print("============================================================================\n")
PY
