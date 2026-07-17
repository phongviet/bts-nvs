#!/usr/bin/env bash
# Week 2, init arm (c) only: train/render/score the VGGT pseudo-cloud init on
# the 5 public scenes (real GT available), using the point clouds already
# built on Kaggle (kaggle_vggt_init.ipynb) and re-integrated locally into
# data/processed/phase1/<scene>/{vggt,colmap_vggt_init,train_staging_vggt}.
#
# Does NOT call build_vggt_init.py -- VGGT inference must not be re-run
# locally (OOMs on the 6GB card; the Kaggle output is already in place).
#
# Fixed baseline backend (vanilla splatfacto, 30k iters -- same as exp001)
# so any score delta vs. arm (a) sparse control is attributable to the init
# alone (docs/plan_week2.md "golden rule": change one axis at a time).
#
# Caveat (see conversation / experiment_log.md): the Kaggle notebook run
# used CHUNK_SIZE=4 for the per-chunk Umeyama alignment (recommended: 16-48),
# and per-chunk fitted scale varied 3.9x-30.6x across the run's 722 chunks --
# a sign of poorly-conditioned alignment. Treat this arm's scores as a
# provisional datapoint, not a clean read on VGGT's ceiling.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airace

PROCESSED_ROOT=data/processed/phase1
OUT_ROOT=runs/phase1/exp003_vggt_init
RESULTS_CSV=results/week2_init_ablation.csv

SCENES=(hcm0031 hcm0034 HCM0181 HCM0193 HCM0204)
SPLIT=public_set
MAX_ITERS=30000

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
  scene_dir="data/raw/phase1/$SPLIT/$scene"
  staging="$PROCESSED_ROOT/$scene/train_staging_vggt"
  run_dir="$OUT_ROOT/$scene"

  if [ ! -d "$staging" ]; then
    echo "!! $scene: $staging missing -- re-integrate the Kaggle VGGT output first. Skipping."
    continue
  fi

  echo "=========================================="
  echo "== $scene (VGGT init) =="
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

  python src/metrics.py --renders "$run_dir/renders_test" \
    --gt "$scene_dir/test/images" --out "$run_dir/metrics_val.json"

  n=$(n_points_of "$PROCESSED_ROOT/$scene/colmap_vggt_init")
  append_row "$scene" "c_vggt" "$n" "$run_dir/metrics_val.json"

  echo "== $scene done =="
done

echo "VGGT-init run on public scenes complete. Results appended to $RESULTS_CSV"
column -s, -t "$RESULTS_CSV"
