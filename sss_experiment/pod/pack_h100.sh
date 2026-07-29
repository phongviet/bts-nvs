#!/usr/bin/env bash
# Build the self-contained scp bundle for the H100 full-pipeline run.
# Layout (unpacks to the pod's $HOME/h100_run/):
#   3D-student-splatting-and-scooping/   SSS repo (patched)
#   radegs/                              RaDe-GS repo (+ export_depth, render_test_depth_csv)
#   bts-nvs/{Analysis/{04,10}.py, src/{render,metrics}.py,
#            data/raw/VAI_NVS_DATA_ROUND2/<scene>/test/test_poses.csv}
#   data/<scene>/{train (raw SIMPLE_RADIAL), test_poses.csv}
#   *.py + *.sh tools at the root
#
#   usage:  ./pack_h100.sh <scene>            # scene = HCM0674
set -euo pipefail
cd "$(dirname "$0")/.."                         # -> sss_experiment/
SCENE=${1:?usage: pack_h100.sh <scene>}
BTS=../bts-nvs
RAW=$BTS/data/raw/VAI_NVS_DATA_ROUND2
RADEGS_SRC=../depth_experiment/repos/radegs
STAGE=$(mktemp -d)/h100_run
mkdir -p "$STAGE/data/$SCENE" "$STAGE/bts-nvs/Analysis" "$STAGE/bts-nvs/src"

[ -d "$RAW/$SCENE/train" ] || { echo "FATAL: missing $RAW/$SCENE/train"; exit 1; }
[ -f "$RAW/$SCENE/test/test_poses.csv" ] || { echo "FATAL: missing test_poses.csv"; exit 1; }

echo "== SSS repo =="
rsync -a --exclude='.git' --exclude='**/build/' --exclude='**/__pycache__/' \
  --exclude='**/*.egg-info' 3D-student-splatting-and-scooping "$STAGE/"

echo "== RaDe-GS repo (with exporters) =="
rsync -a --exclude='.git' --exclude='**/build/' --exclude='**/__pycache__/' \
  --exclude='**/*.egg-info' "$RADEGS_SRC/" "$STAGE/radegs/"
# sanity: the two exporters must be present
for f in export_depth.py render_test_depth_csv.py; do
  [ -f "$STAGE/radegs/$f" ] || { echo "FATAL: radegs/$f missing"; exit 1; }
done

echo "== scene data (raw train + test poses) =="
rsync -a "$RAW/$SCENE/train" "$STAGE/data/$SCENE/"
cp "$RAW/$SCENE/test/test_poses.csv" "$STAGE/data/$SCENE/test_poses.csv"

echo "== bts-nvs refiner subset (no nerfstudio needed) =="
cp "$BTS/Analysis/04_x3_dibr_pilot.py" "$BTS/Analysis/10_refiner_pilot.py" "$STAGE/bts-nvs/Analysis/"
cp "$BTS/src/render.py" "$BTS/src/metrics.py" "$STAGE/bts-nvs/src/"
mkdir -p "$STAGE/bts-nvs/data/raw/VAI_NVS_DATA_ROUND2/$SCENE/test"
cp "$RAW/$SCENE/test/test_poses.csv" \
   "$STAGE/bts-nvs/data/raw/VAI_NVS_DATA_ROUND2/$SCENE/test/test_poses.csv"

echo "== tools + scripts =="
cp render_test_csv.py make_undistorted_scene.py "$STAGE/"
cp pod/make_train_poses_csv.py pod/provision_pod_h100.sh pod/run_full_pipeline_h100.sh \
   pod/gpu_monitor.sh "$STAGE/"
chmod +x "$STAGE"/*.sh

OUT=$PWD/h100_run_$SCENE.tar.gz
tar -C "$(dirname "$STAGE")" -czf "$OUT" h100_run
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "top-level:"; tar -tzf "$OUT" | sed 's,^h100_run/,,' | awk -F/ 'NF>1{print $1"/"$2}' | sort -u | head -30
rm -rf "$(dirname "$STAGE")"
echo ""
echo "NEXT: scp to pod, then on the pod:"
echo "  tar -C \$HOME -xzf h100_run_$SCENE.tar.gz"
echo "  cd \$HOME/h100_run && ./provision_pod_h100.sh 2>&1 | tee provision.log"
echo "  ./run_full_pipeline_h100.sh $SCENE"
echo "  # download: scp -r pod:h100_run/out/$SCENE ./"
