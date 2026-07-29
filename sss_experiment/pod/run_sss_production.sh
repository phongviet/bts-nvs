#!/usr/bin/env bash
# PRODUCTION SSS run for a single round-2 scene on a rented RTX 4090.
#
# Full-quality, submission-grade SSS backbone: trains on the FULL train set
# (not a holdout), renders the official hidden test poses, and KEEPS the .ply
# so the model can be downloaded and re-rendered locally without re-training.
#
#   usage:  ./run_sss_production.sh <scene>          # scene = bonsai | HCM0674
#
# Quality knobs (user spec): 60k iterations, cap_max 4,000,000 comps, full
# resolution (-r 1 -- bonsai is 1920x1080; the >1600 auto-downscale would drop
# it to 1600 and mismatch the full-res test poses). nu_degree/C/C_burnin/burnin
# are the paper-scale SSS params proven on the fleet.
#
# data_device cpu: 240 full-res images + 4M comps will not co-reside with the
# image cache in 24 GB otherwise.
#
# SGHMC robustness: the post-burnin noise phase can (rarely, stochastically)
# crash the rasterizer. Rolling checkpoints every 2500 iters + a 4-attempt
# auto-resume re-rolls the noise from the last checkpoint instead of losing the
# whole run. The final ply is KEPT (this is the deliverable); interim
# checkpoints are pruned once the ply lands.
set -uo pipefail

SCENE=${1:?usage: run_sss_production.sh <scene>}
ROOT=${ROOT:-$HOME/sss_prod}
SSS=$ROOT/3D-student-splatting-and-scooping
RAWSRC=$ROOT/data/$SCENE/train                    # as-shipped COLMAP (may be SIMPLE_RADIAL)
SRC=$ROOT/data/$SCENE/train_pinhole               # undistorted, what SSS actually trains on
POSES=$ROOT/data/$SCENE/test_poses.csv            # official hidden test poses
TRAINPOSES=$ROOT/data/$SCENE/train_poses.csv      # derived from train_pinhole (DIBR sources)
OUT=$ROOT/out/$SCENE
MODEL=$OUT/model
RENDERS=$OUT/sss_renders                          # test + train pooled -> --render-override
LOG=$OUT/train.log
ITERS=${ITERS:-60000}
CAP=${CAP:-4000000}
mkdir -p "$OUT"

[ -d "$RAWSRC" ]   || { echo "FATAL: missing train dir $RAWSRC"; exit 1; }
[ -f "$POSES" ]    || { echo "FATAL: missing test poses $POSES"; exit 1; }

# Activate the sss conda env by ABSOLUTE path: this script runs in a non-login
# shell (from pipeline.sh) where conda is not on PATH, so `conda info --base`
# would fail and -- without set -e -- silently leave us on the base-image python
# (which has torch but NOT cv2 or the SSS CUDA extensions). Hardcode it.
#
# DOUBLE-ACTIVATION TRAP (Jul-23, cost one failed launch): if the CALLER already
# did `conda activate sss`, we inherit CONDA_SHLVL/CONDA_DEFAULT_ENV=sss, so the
# `conda activate sss` below is a NO-OP -- while the unconditional PATH export
# above has just shoved BASE's /opt/miniconda/bin in FRONT of the env's bin dir.
# `python` then resolves to the BASE interpreter, which has torch but no cv2, and
# the dep check fails with a completely misleading "No module named 'cv2'".
# Fix: drop any inherited activation first, then activate for real, then ASSERT
# the interpreter actually lives in the env prefix.
export PATH=/opt/miniconda/bin:$PATH
source /opt/miniconda/etc/profile.d/conda.sh
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate sss || { echo "FATAL: could not activate conda env 'sss'"; exit 1; }
case "$(command -v python)" in
  "$CONDA_PREFIX"/bin/python) : ;;
  *) echo "FATAL: python is $(command -v python), not $CONDA_PREFIX/bin/python -- PATH is wrong"; exit 1 ;;
esac
python -c "import cv2, torch, diff_t_rasterization, simple_knn" \
  || { echo "FATAL: sss env missing deps (cv2/torch/rasterizer) -- provision incomplete"; exit 1; }
export PYTHONPATH="$SSS:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 0. undistort to pinhole (SSS/3DGS rasterizer ignores radial k; SIMPLE_RADIAL
#        drone scenes MUST be undistorted or images and geometry disagree. K is
#        kept unchanged, mirroring nerfstudio, so the test poses still apply.
#        bonsai is SIMPLE_PINHOLE -> k=0 -> byte-identical passthrough copy). ---
if [ ! -f "$SRC/sparse/0/cameras.bin" ]; then
  echo "== [$SCENE] undistort -> pinhole [$(date +%H:%M)] =="
  python "$ROOT/make_undistorted_scene.py" --src "$RAWSRC" --dst "$SRC"
fi
python - "$SRC/sparse/0/cameras.bin" <<'PY'
import struct, sys
m = struct.unpack("<i", open(sys.argv[1],"rb").read()[12:16])[0]
name = {0:"SIMPLE_PINHOLE",1:"PINHOLE"}.get(m)
assert m in (0,1), f"train_pinhole still non-pinhole (model {m}) -- undistort failed"
print(f"   train_pinhole camera model: {name} (OK)")
PY

ply_glob() { ls "$MODEL"/point_cloud/iteration_*/point_cloud.ply 2>/dev/null | tail -1; }

# --- 1. train SSS (full res, 4-attempt auto-resume) ---
CKPTS=(); for i in $(seq 2500 2500 $((ITERS-1))); do CKPTS+=("$i"); done
if [ -z "$(ply_glob)" ]; then
  for attempt in 1 2 3 4; do
    [ -n "$(ply_glob)" ] && break
    echo "== [$SCENE] SSS train attempt $attempt [$(date +%H:%M)] iters=$ITERS cap=$CAP res=1 =="
    RESUME=(); last=$(ls "$MODEL"/chkpnt*.pth 2>/dev/null | sed 's/[^0-9]*\([0-9]*\).pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2- || true)
    [ -n "$last" ] && { echo "   resuming from $last"; RESUME=(--start_checkpoint "$last"); }
    set +e
    python "$SSS/train.py" -s "$SRC" -m "$MODEL" -r 1 --data_device cpu \
      --cap_max "$CAP" --nu_degree 100 --C_burnin 5e5 --C 1.2e2 \
      --burnin_iterations 7000 --iterations "$ITERS" \
      --save_iterations "$ITERS" --test_iterations "$ITERS" --quiet \
      --checkpoint_iterations "${CKPTS[@]}" "${RESUME[@]}" \
      >>"$LOG" 2>&1
    rc=$?; set -e
    echo "   attempt $attempt rc=$rc [$(date +%H:%M)]"
    if [ "$attempt" = 4 ] && [ -z "$(ply_glob)" ]; then
      echo "FATAL: [$SCENE] SSS train failed 4x -- tail of $LOG:"; tail -40 "$LOG"; exit 1
    fi
  done
  rm -f "$MODEL"/chkpnt*.pth   # multi-GB; the ply is the deliverable
fi
echo "== [$SCENE] trained ply: $(ply_glob) ($(du -h "$(ply_glob)" | cut -f1)) =="
touch "$OUT/.done_ply"          # watcher pulls the ply NOW, while stage 2 renders

# --- 2. render TEST + TRAIN poses into ONE pool ---
# The local refiner's --render-override needs BOTH: test poses are the frames we
# ship, train poses are the DIBR source views it learns from. bonsai's shipped
# override pool (sss_renders_ho) is exactly 220 train + 28 test = 248 files, so
# rendering only the test poses here would leave the pool unusable.
# nu_degree MUST match training (=100).
python "$ROOT/render_test_csv.py" --model "$MODEL" \
  --poses-csv "$POSES" --out "$RENDERS" --nu-degree 100
n_test=$(ls "$RENDERS" 2>/dev/null | wc -l)

[ -f "$TRAINPOSES" ] || python "$ROOT/make_train_poses_csv.py" --scene-dir "$SRC" --out "$TRAINPOSES"
python "$ROOT/render_test_csv.py" --model "$MODEL" \
  --poses-csv "$TRAINPOSES" --out "$RENDERS" --nu-degree 100
n_all=$(ls "$RENDERS" 2>/dev/null | wc -l)
echo "== [$SCENE] render pool: $n_all files ($n_test test + $((n_all - n_test)) train) -> $RENDERS =="
[ "$n_all" -gt "$n_test" ] || { echo "FATAL: train poses rendered nothing"; exit 1; }
touch "$OUT/.done_renders"

echo "== [$SCENE] PRODUCTION RUN COMPLETE. Deliverables under $OUT:"
echo "     model/point_cloud/iteration_${ITERS}/point_cloud.ply   (re-renderable)"
echo "     sss_renders/                                           (test + train pose renders)"
echo "     train.log"
touch "$OUT/.done_all"
