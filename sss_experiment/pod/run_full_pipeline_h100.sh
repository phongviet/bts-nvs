#!/usr/bin/env bash
# FULL max-quality pipeline for one round-2 scene on a rented H100 80GB.
#
#   SSS (100k iter, 32M cap, scaled hyperparams)  -> test+train renders + ply
#   RaDe-GS (30k iter)                            -> all DIBR depth (train+test)
#   Refiner (COLMAP-native: SSS render backbone + RaDe-GS depth, no splatfacto)
#                                                 -> refined test renders + net
# Continuous GPU telemetry is logged for the whole run.
#
#   usage:  ./run_full_pipeline_h100.sh <scene>          # scene = HCM0674 | ...
#
# Every model + render + depth + the GPU CSV is kept under $ROOT/out/<scene>/ so
# the whole thing can be scp'd home and re-rendered without re-training.
#
# VRAM NOTE: cap_max 32M at ~2.5 KB/comp is ~74 GB of gaussian+optimizer state
# plus overhead -- fits an 80 GB H100 but with a thin margin near the end of
# densification. The GPU CSV captures the real peak; if it OOMs, re-run with
# CAP=24000000 (env override). SGHMC crashes auto-resume from the last ckpt.
set -uo pipefail

SCENE=${1:?usage: run_full_pipeline_h100.sh <scene>}
ROOT=${ROOT:-$HOME/h100_run}
SSS=$ROOT/3D-student-splatting-and-scooping
RADEGS=$ROOT/radegs
BTS=$ROOT/bts-nvs

RAWSRC=$ROOT/data/$SCENE/train                 # as-shipped COLMAP (SIMPLE_RADIAL)
PIN=$ROOT/data/$SCENE/train_pinhole            # undistorted pinhole (SSS/RaDe-GS train on this)
TESTPOSES=$ROOT/data/$SCENE/test_poses.csv     # official hidden test poses
TRAINPOSES=$ROOT/data/$SCENE/train_poses.csv   # built from PIN (refiner training-pair poses)

OUT=$ROOT/out/$SCENE
SSS_MODEL=$OUT/sss_model
SSS_RENDERS=$OUT/sss_renders                   # test + train renders (render_override pool)
RADEGS_MODEL=$OUT/radegs_model
RADEGS_DEPTH=$OUT/radegs_depth                 # train + test depth (depth_source + depth_T)
GPUCSV=$OUT/gpu_metrics.csv
LOG=$OUT/pipeline.log

# quality knobs (env-overridable)
ITERS=${ITERS:-100000}
CAP=${CAP:-32000000}
RADEGS_ITERS=${RADEGS_ITERS:-30000}
REF_ITERS=${REF_ITERS:-3000}
mkdir -p "$OUT"

exec > >(tee -a "$LOG") 2>&1
echo "======================================================================"
echo "== H100 FULL PIPELINE  scene=$SCENE  [$(date)]"
echo "==   SSS iters=$ITERS cap=$CAP | RaDe-GS iters=$RADEGS_ITERS | refiner iters=$REF_ITERS"
echo "======================================================================"

[ -d "$RAWSRC" ] || { echo "FATAL: missing $RAWSRC"; exit 1; }
[ -f "$TESTPOSES" ] || { echo "FATAL: missing $TESTPOSES"; exit 1; }

# --- conda env activation by ABSOLUTE path (non-login shell; see gotcha notes) ---
export PATH=/opt/miniconda/bin:$PATH
source /opt/miniconda/etc/profile.d/conda.sh
act() { conda activate "$1" || { echo "FATAL: cannot activate env '$1'"; exit 1; }; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- start GPU telemetry for the whole run ---
bash "$ROOT/gpu_monitor.sh" "$GPUCSV" 5 &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT
echo "== GPU telemetry -> $GPUCSV (pid $MON) =="

ply_glob() { ls "$SSS_MODEL"/point_cloud/iteration_*/point_cloud.ply 2>/dev/null | tail -1; }

# ======================================================================
# Stage 0 — undistort SIMPLE_RADIAL -> pinhole (+ filter images.bin to train) &
#           build train_poses.csv from the pinhole scene
# ======================================================================
act sss
python -c "import cv2, torch, diff_t_rasterization, simple_knn" \
  || { echo "FATAL: sss env incomplete"; exit 1; }
if [ ! -f "$PIN/sparse/0/cameras.bin" ]; then
  echo "== [$SCENE] Stage 0: undistort -> pinhole [$(date +%H:%M)] =="
  python "$ROOT/make_undistorted_scene.py" --src "$RAWSRC" --dst "$PIN"
fi
python - "$PIN/sparse/0/cameras.bin" <<'PY'
import struct, sys
m = struct.unpack("<i", open(sys.argv[1],"rb").read()[12:16])[0]
assert m in (0,1), f"train_pinhole still non-pinhole (model {m})"
print(f"   train_pinhole camera model OK (id {m})")
PY
[ -f "$TRAINPOSES" ] || python "$ROOT/make_train_poses_csv.py" --scene-dir "$PIN" --out "$TRAINPOSES"

# ======================================================================
# Stage 1 — SSS train (100k, 32M cap, scaled schedule, 4-attempt auto-resume)
# ======================================================================
export PYTHONPATH="$SSS:${PYTHONPATH:-}"
# Checkpoint every CKPT_EVERY iters for SGHMC crash-resume. At 32M cap each
# chkpnt is many GB, so a coarse interval (20k) bounds peak disk to a few live
# checkpoints; a crash re-does at most CKPT_EVERY iters. Pruned once the ply lands.
CKPT_EVERY=${CKPT_EVERY:-20000}
CKPTS=(); for i in $(seq "$CKPT_EVERY" "$CKPT_EVERY" $((ITERS-1))); do CKPTS+=("$i"); done
if [ -z "$(ply_glob)" ]; then
  for attempt in 1 2 3 4; do
    [ -n "$(ply_glob)" ] && break
    echo "== [$SCENE] Stage 1: SSS train attempt $attempt [$(date +%H:%M)] iters=$ITERS cap=$CAP =="
    RESUME=(); last=$(ls "$SSS_MODEL"/chkpnt*.pth 2>/dev/null | sed 's/[^0-9]*\([0-9]*\).pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2- || true)
    [ -n "$last" ] && { echo "   resuming from $last"; RESUME=(--start_checkpoint "$last"); }
    set +e
    # scaled schedule for 100k/32M: longer burnin, long densify window to grow
    # into the 32M cap, position-lr decay matched to the full run length.
    python "$SSS/train.py" -s "$PIN" -m "$SSS_MODEL" -r 1 --data_device cpu \
      --cap_max "$CAP" --nu_degree 100 --C_burnin 5e5 --C 1.2e2 \
      --burnin_iterations 12000 --iterations "$ITERS" \
      --densify_until_iter 50000 --position_lr_max_steps "$ITERS" \
      --save_iterations "$ITERS" --test_iterations "$ITERS" --quiet \
      --checkpoint_iterations "${CKPTS[@]}" "${RESUME[@]}"
    rc=$?; set -e
    echo "   attempt $attempt rc=$rc [$(date +%H:%M)]"
    if [ "$attempt" = 4 ] && [ -z "$(ply_glob)" ]; then
      echo "FATAL: SSS train failed 4x"; exit 1
    fi
  done
  rm -f "$SSS_MODEL"/chkpnt*.pth
fi
echo "== [$SCENE] SSS ply: $(ply_glob) ($(du -h "$(ply_glob)" | cut -f1)) =="

# ======================================================================
# Stage 2 — SSS renders at TEST + TRAIN poses (one pool for render_override)
# ======================================================================
echo "== [$SCENE] Stage 2: SSS render test+train poses [$(date +%H:%M)] =="
python "$ROOT/render_test_csv.py" --model "$SSS_MODEL" --poses-csv "$TESTPOSES" \
  --out "$SSS_RENDERS" --nu-degree 100
python "$ROOT/render_test_csv.py" --model "$SSS_MODEL" --poses-csv "$TRAINPOSES" \
  --out "$SSS_RENDERS" --nu-degree 100
echo "   SSS renders: $(ls "$SSS_RENDERS" | wc -l) (test+train) -> $SSS_RENDERS"

# ======================================================================
# Stage 3 — RaDe-GS train (30k)
# ======================================================================
act gs
python -c "import torch, diff_gaussian_rasterization, simple_knn" \
  || { echo "FATAL: gs (radegs) env incomplete"; exit 1; }
if [ ! -f "$RADEGS_MODEL/point_cloud/iteration_${RADEGS_ITERS}/point_cloud.ply" ]; then
  echo "== [$SCENE] Stage 3: RaDe-GS train iters=$RADEGS_ITERS [$(date +%H:%M)] =="
  python "$RADEGS/train.py" -s "$PIN" -m "$RADEGS_MODEL" -r 1 \
    --iterations "$RADEGS_ITERS" --save_iterations "$RADEGS_ITERS" \
    --test_iterations "$RADEGS_ITERS" --quiet
fi

# ======================================================================
# Stage 4 — RaDe-GS depth at TRAIN views + TEST poses (one dir)
# ======================================================================
echo "== [$SCENE] Stage 4: RaDe-GS export depth (train + test) [$(date +%H:%M)] =="
python "$RADEGS/export_depth.py" -m "$RADEGS_MODEL" -s "$PIN" -r 1 \
  --iteration "$RADEGS_ITERS" --depth_out "$RADEGS_DEPTH"
python "$RADEGS/render_test_depth_csv.py" -m "$RADEGS_MODEL" -s "$PIN" -r 1 \
  --iteration "$RADEGS_ITERS" --poses-csv "$TESTPOSES" --depth_out "$RADEGS_DEPTH"
echo "   RaDe-GS depth maps: $(ls "$RADEGS_DEPTH"/*.npy | wc -l) (train+test) -> $RADEGS_DEPTH"

# ======================================================================
# Stage 5 — Refiner (COLMAP-native: SSS render + RaDe-GS depth, no splatfacto)
# ======================================================================
act sss
echo "== [$SCENE] Stage 5: COLMAP-native refiner [$(date +%H:%M)] =="
cd "$BTS/Analysis"
python 10_refiner_pilot.py --scene "$SCENE" \
  --render-override "$SSS_RENDERS" \
  --depth-source "$RADEGS_DEPTH" \
  --target-depth-source "$RADEGS_DEPTH" \
  --train-dir "$PIN" \
  --iters "$REF_ITERS" --base 32 --png --suffix _cn
REF_OUT="$BTS/Analysis/X5_refiner/$SCENE"
echo "== [$SCENE] refiner outputs: $REF_OUT (refiner_cn.pt + renders_refined_cn/) =="

# ======================================================================
# Collect deliverables under $OUT for a single scp home
# ======================================================================
echo "== [$SCENE] collecting deliverables [$(date +%H:%M)] =="
mkdir -p "$OUT/refiner"
cp -f "$REF_OUT/refiner_cn.pt" "$OUT/refiner/" 2>/dev/null || true
cp -rf "$REF_OUT/renders_refined_cn" "$OUT/refiner/" 2>/dev/null || true
kill $MON 2>/dev/null || true
cat > "$OUT/MANIFEST.txt" <<EOF
scene=$SCENE  finished=$(date)
sss_model/point_cloud/iteration_${ITERS}/point_cloud.ply   SSS backbone ($CAP cap)
sss_renders/                                               SSS renders (test + train)
radegs_model/point_cloud/iteration_${RADEGS_ITERS}/        RaDe-GS model
radegs_depth/                                              RaDe-GS depth (train + test .npy)
refiner/renders_refined_cn/                                FINAL refined test renders (PNG)
refiner/refiner_cn.pt                                      trained refiner weights
gpu_metrics.csv                                            full-run GPU telemetry
pipeline.log                                               this run's log
EOF
echo "======================================================================"
echo "== [$SCENE] PIPELINE COMPLETE. Deliverables under $OUT :"
cat "$OUT/MANIFEST.txt"
echo "== peak VRAM: $(awk -F, 'NR>1{if($3>m)m=$3}END{print m" MiB"}' "$GPUCSV" 2>/dev/null) =="
echo "======================================================================"
