#!/usr/bin/env bash
# Week 2, init arm (b) only: train/render(/score where GT exists) the
# dense-COLMAP init, using point clouds built by src/data_prep/build_dense_colmap.py
# (locally, or on Kaggle via kaggle-dense-colmap-init.ipynb + re-integrated
# into data/processed/phase1/<scene>/{dense,colmap_dense_init,train_staging_dense}).
#
# Runs on EVERY scene that currently has train_staging_dense ready (discovered
# dynamically -- not a fixed list), across both public_set and private_set1.
# Public scenes have local test GT (test/images/) so we score them into
# results/week2_init_ablation.csv; private scenes only have test_poses.csv
# (no GT shipped), so we still train+render (useful groundwork for later
# submission packaging) but skip scoring for those.
#
# Does NOT call build_dense_colmap.py -- run that (or the Kaggle notebook)
# first for any scene missing train_staging_dense; this script only trains.
#
# Fixed baseline backend (vanilla splatfacto, 30k iters -- same as exp001 and
# exp003_vggt_init) so any score delta vs. arm (a) sparse control is
# attributable to the init alone (docs/plan_week2.md "golden rule": change
# one axis at a time).
#
# Context: arm (c) VGGT was scored on all 5 public scenes and lost to the
# sparse control on every one (-0.009 to -0.022 score, see
# results/week2_init_ablation.csv), consistent with the chunk_size=4
# alignment-instability caveat documented in run_week2_vggt_public.sh.
# Dense COLMAP is the remaining candidate to beat the sparse control.
# Its raw fused point clouds carry a small (<1%) tail of far stray points
# from stereo-fusion noise -- checked before this run and judged harmless
# (1st-99th percentile extent tracks the sparse-COLMAP extent; see
# conversation / experiment_log.md).
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airace

PROCESSED_ROOT=data/processed/phase1
RAW_ROOT=data/raw/phase1
OUT_ROOT=runs/phase1/exp002_dense_colmap_init
RESULTS_CSV=results/week2_init_ablation.csv
MAX_ITERS=30000

# Discover every scene with a ready dense-COLMAP staging dir, across both splits.
SCENES=()
for split_dir in "$RAW_ROOT"/public_set "$RAW_ROOT"/private_set1; do
  [ -d "$split_dir" ] || continue
  for scene_dir in "$split_dir"/*/; do
    scene=$(basename "$scene_dir")
    if [ -d "$PROCESSED_ROOT/$scene/train_staging_dense" ]; then
      SCENES+=("$scene")
    fi
  done
done
echo "Scenes with dense-COLMAP init ready: ${SCENES[*]}"

if [ ! -f "$RESULTS_CSV" ]; then
  echo "scene,arm,n_points,psnr,ssim,lpips,score,timestamp,git_commit" > "$RESULTS_CSV"
fi

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
  python - "$1" <<'PYEOF'
import sys
from nerfstudio.data.utils.colmap_parsing_utils import read_points3D_binary
print(len(read_points3D_binary(sys.argv[1] + "/points3D.bin")))
PYEOF
}

for scene in "${SCENES[@]}"; do
  staging="$PROCESSED_ROOT/$scene/train_staging_dense"
  run_dir="$OUT_ROOT/$scene"

  if [ -d "$RAW_ROOT/public_set/$scene" ]; then
    scene_dir="$RAW_ROOT/public_set/$scene"
    has_gt=1
  else
    scene_dir="$RAW_ROOT/private_set1/$scene"
    has_gt=0
  fi

  echo "=========================================="
  echo "== $scene (dense-COLMAP init) =="
  echo "=========================================="

  if [ -d "$run_dir" ]; then
    echo "== $scene: run_dir exists, skipping training =="
  else
    ns-train splatfacto \
      --data "$staging" \
      --output-dir "$run_dir" \
      --max-num-iterations "$MAX_ITERS" \
      --viewer.quit-on-train-completion True \
      colmap --eval-mode all --colmap-path sparse/0
  fi

  config=$(find "$run_dir" -name config.yml | sort | tail -1)

  if [ ! -f "$run_dir/renders_test/.done" ]; then
    python src/render.py --config "$config" --mode test \
      --poses-csv "$scene_dir/test/test_poses.csv" --out "$run_dir/renders_test"
    touch "$run_dir/renders_test/.done"
  fi

  if [ "$has_gt" -eq 1 ]; then
    python src/metrics.py --renders "$run_dir/renders_test" \
      --gt "$scene_dir/test/images" --out "$run_dir/metrics_val.json"

    n=$(n_points_of "$PROCESSED_ROOT/$scene/colmap_dense_init")
    append_row "$scene" "b_dense" "$n" "$run_dir/metrics_val.json"
  else
    echo "== $scene: private scene, no local test GT -- rendered only, skipping scoring =="
  fi

  echo "== $scene done =="
done

echo "Dense-COLMAP-init run complete. Results appended to $RESULTS_CSV"
column -s, -t "$RESULTS_CSV"
