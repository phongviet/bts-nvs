#!/usr/bin/env bash
# Week 2, Day 1-3: init-arm ablation (sparse control / dense COLMAP / VGGT
# pseudo-cloud) on the 5 public scenes (real GT available), fixed baseline
# backend (vanilla splatfacto, 30k iters -- same as exp001) so any score
# delta is attributable to the init arm alone (docs/plan_week2.md "golden
# rule": change one axis at a time).
#
# Arm (a) sparse control reuses the existing exp001 run/metrics if present
# (identical backend -> no retraining needed).
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airace

PROCESSED_ROOT=data/processed/phase1
OUT_ROOT=runs/phase1
RESULTS_CSV=results/week2_init_ablation.csv
BASELINE_RUN_ROOT=$OUT_ROOT/exp001_baseline_splatfacto

SCENES=(hcm0031 hcm0034 HCM0181 HCM0193 HCM0204)
SPLIT=public_set
MAX_ITERS=30000

if [ ! -f "$RESULTS_CSV" ]; then
  echo "scene,arm,n_points,psnr,ssim,lpips,score,timestamp,git_commit" > "$RESULTS_CSV"
fi

git_commit() { git rev-parse --short HEAD 2>/dev/null || echo "PENDING"; }

append_row() {
  local scene=$1 arm=$2 npts=$3 metrics_json=$4
  python - "$scene" "$arm" "$npts" "$metrics_json" "$RESULTS_CSV" <<'PYEOF'
import json, sys, datetime
scene, arm, npts, metrics_json, csv_path = sys.argv[1:6]
d = json.load(open(metrics_json))
per = d["per_image"]
n = len(per)
mean = lambda k: sum(x[k] for x in per) / n
row = f"{scene},{arm},{npts},{mean('psnr'):.4f},{mean('ssim'):.4f},{mean('lpips'):.4f},{mean('score'):.4f},{datetime.datetime.now().isoformat(timespec='seconds')},PENDING\n"
with open(csv_path, "a") as f:
    f.write(row)
print(row.strip())
PYEOF
}

n_points_of() {
  # arg: sparse dir with points3D.bin
  python - "$1" <<'PYEOF'
import sys
from nerfstudio.data.utils.colmap_parsing_utils import read_points3D_binary
print(len(read_points3D_binary(sys.argv[1] + "/points3D.bin")))
PYEOF
}

train_and_score() {
  local scene=$1 arm=$2 staging=$3 run_dir=$4 scene_dir=$5

  if [ -d "$run_dir" ]; then
    echo "== $scene/$arm: run_dir exists, skipping training =="
  else
    echo "== $scene/$arm: training splatfacto ($MAX_ITERS iters) =="
    ns-train splatfacto \
      --data "$staging" \
      --output-dir "$run_dir" \
      --max-num-iterations "$MAX_ITERS" \
      --viewer.quit-on-train-completion True \
      colmap --eval-mode all --colmap-path sparse/0
  fi

  local config
  config=$(find "$run_dir" -name config.yml | sort | tail -1)

  if [ ! -f "$run_dir/renders_test/.done" ]; then
    python src/render.py --config "$config" --mode test \
      --poses-csv "$scene_dir/test/test_poses.csv" --out "$run_dir/renders_test"
    touch "$run_dir/renders_test/.done"
  fi

  python src/metrics.py --renders "$run_dir/renders_test" \
    --gt "$scene_dir/test/images" --out "$run_dir/metrics_val.json"
}

for scene in "${SCENES[@]}"; do
  scene_dir="data/raw/phase1/$SPLIT/$scene"

  # --- Arm (a): sparse control (reuse exp001 if it exists) ---
  a_run="$BASELINE_RUN_ROOT/$scene"
  if [ -f "$a_run/metrics_val.json" ]; then
    echo "== $scene/a_sparse: reusing exp001 metrics =="
    n=$(n_points_of "$PROCESSED_ROOT/$scene/colmap_train_only")
    append_row "$scene" "a_sparse" "$n" "$a_run/metrics_val.json"
  else
    train_and_score "$scene" "a_sparse" "$PROCESSED_ROOT/$scene/train_staging" "$a_run" "$scene_dir"
    n=$(n_points_of "$PROCESSED_ROOT/$scene/colmap_train_only")
    append_row "$scene" "a_sparse" "$n" "$a_run/metrics_val.json"
  fi

  # --- Arm (b): dense COLMAP init ---
  echo "== $scene/b_dense: building dense COLMAP init =="
  python src/data_prep/build_dense_colmap.py \
    --scene-dir "$scene_dir" --processed-root "$PROCESSED_ROOT" \
    --max-points 2000000 --max-image-size 1600 --gpu-index 0
  b_run="$OUT_ROOT/exp002_dense_colmap_init/$scene"
  train_and_score "$scene" "b_dense" "$PROCESSED_ROOT/$scene/train_staging_dense" "$b_run" "$scene_dir"
  n=$(n_points_of "$PROCESSED_ROOT/$scene/colmap_dense_init")
  append_row "$scene" "b_dense" "$n" "$b_run/metrics_val.json"

  # --- Arm (c): VGGT pseudo-cloud init ---
  echo "== $scene/c_vggt: building VGGT pseudo-cloud init =="
  python src/data_prep/build_vggt_init.py \
    --scene-dir "$scene_dir" --processed-root "$PROCESSED_ROOT" \
    --chunk-size 16 --conf-percentile 50 --points-per-frame-cap 8000 --max-points 2000000
  c_run="$OUT_ROOT/exp003_vggt_init/$scene"
  train_and_score "$scene" "c_vggt" "$PROCESSED_ROOT/$scene/train_staging_vggt" "$c_run" "$scene_dir"
  n=$(n_points_of "$PROCESSED_ROOT/$scene/colmap_vggt_init")
  append_row "$scene" "c_vggt" "$n" "$c_run/metrics_val.json"

  echo "== $scene: all arms done =="
done

echo "Week 2 init ablation complete. Results: $RESULTS_CSV"
column -s, -t "$RESULTS_CSV"
