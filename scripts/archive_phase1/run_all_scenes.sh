#!/usr/bin/env bash
# Runs the full exp001 pipeline (filter -> train -> render -> metrics/package)
# for every remaining scene in configs/experiments/exp001_baseline_splatfacto.yaml.
# hcm0034 is already done and is skipped automatically.
set -euo pipefail
cd "$(dirname "$0")/.."

EXP=exp001_baseline_splatfacto
PROCESSED_ROOT=data/processed/phase1
OUT_ROOT=runs/phase1/$EXP

declare -A SPLIT_OF=(
  [hcm0031]=public_set [hcm0034]=public_set [HCM0181]=public_set
  [HCM0193]=public_set [HCM0204]=public_set
  [HCM0249]=private_set1 [HCM0254]=private_set1 [HCM0276]=private_set1
  [HCM1439]=private_set1 [HNI0131]=private_set1 [HNI0265]=private_set1
  [HNI0366]=private_set1 [HNI0437]=private_set1
)

for scene in "${!SPLIT_OF[@]}"; do
  split=${SPLIT_OF[$scene]}
  scene_dir=data/raw/phase1/$split/$scene
  staging=$PROCESSED_ROOT/$scene/train_staging
  run_dir=$OUT_ROOT/$scene

  if [ -d "$run_dir" ]; then
    echo "== $scene: run_dir already exists, skipping =="
    continue
  fi

  echo "=========================================="
  echo "== $scene ($split) =="
  echo "=========================================="

  # 1. Filter COLMAP sparse model to train-only images (avoids test leakage)
  python src/data_prep/filter_colmap_train.py \
    --scene-dir "$scene_dir" \
    --processed-root "$PROCESSED_ROOT"

  # 2. Train vanilla splatfacto, 30k iters
  ns-train splatfacto \
    --data "$staging" \
    --output-dir "$run_dir" \
    --max-num-iterations 30000 \
    --viewer.quit-on-train-completion True \
    colmap --eval-mode all --colmap-path sparse/0

  # 3. Find the resulting config.yml (nerfstudio nests by timestamp)
  config=$(find "$run_dir" -name config.yml | sort | tail -1)

  # 4. Render the official test poses
  python src/render.py \
    --config "$config" \
    --mode test \
    --poses-csv "$scene_dir/test/test_poses.csv" \
    --out "$run_dir/renders_test"

  # 5. Metrics only possible where GT test images exist (public_set)
  if [ "$split" = "public_set" ]; then
    python src/metrics.py \
      --renders "$run_dir/renders_test" \
      --gt "$scene_dir/test/images" \
      --out "$run_dir/metrics_val.json"
  fi

  # 6. Package for manual submission
  mkdir -p "submissions/phase1/${EXP}_results/$scene"
  cp "$run_dir"/renders_test/*.JPG "submissions/phase1/${EXP}_results/$scene/" 2>/dev/null || \
  cp "$run_dir"/renders_test/*.jpg "submissions/phase1/${EXP}_results/$scene/"

  echo "== $scene done =="
done

echo "All scenes processed. Results under submissions/phase1/${EXP}_results/<scene>/"
