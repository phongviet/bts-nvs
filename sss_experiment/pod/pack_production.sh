#!/usr/bin/env bash
# Build the self-contained scp bundle for the SSS production pod runs.
# Contains BOTH scenes (bonsai + HCM0674) + a parametrized run script, so the
# same tarball is scp'd to both pods and each runs its own scene:
#   pod1:  ./run_sss_production.sh bonsai
#   pod2:  ./run_sss_production.sh HCM0674
# Produces sss_prod.tar.gz laid out as the pod's $HOME/sss_prod/.
set -euo pipefail
cd "$(dirname "$0")/.."                         # -> sss_experiment/
BTS=../bts-nvs
RAW=$BTS/data/raw/VAI_NVS_DATA_ROUND2
STAGE=$(mktemp -d)/sss_prod
mkdir -p "$STAGE/data"

# SSS repo (patched: simple_knn cfloat + render q98 + SGHMC clamps + eval=False), minus .git/builds
rsync -a --exclude='.git' --exclude='**/build/' --exclude='**/__pycache__/' \
  3D-student-splatting-and-scooping "$STAGE/"

# full train sets (as-shipped COLMAP; bonsai=SIMPLE_PINHOLE, HCM0674=SIMPLE_RADIAL
# -> the run script undistorts on the pod) + official hidden test poses
SCENES=("${@:-bonsai HCM0674}")
[ $# -gt 0 ] && SCENES=("$@")
echo "staging scenes: ${SCENES[*]}"
for SCENE in "${SCENES[@]}"; do
  [ -d "$RAW/$SCENE/train" ] || { echo "FATAL: missing $RAW/$SCENE/train"; exit 1; }
  mkdir -p "$STAGE/data/$SCENE"
  rsync -a "$RAW/$SCENE/train" "$STAGE/data/$SCENE/"
  cp "$RAW/$SCENE/test/test_poses.csv" "$STAGE/data/$SCENE/test_poses.csv"
  echo "  staged $SCENE: $(ls "$STAGE/data/$SCENE/train/images" | wc -l) imgs, "\
"$(wc -l < "$STAGE/data/$SCENE/test_poses.csv") pose-lines"
done

# tools + scripts
cp render_test_csv.py make_undistorted_scene.py "$STAGE/"
cp pod/make_train_poses_csv.py "$STAGE/"        # run_sss_production stage 2 needs it
cp pod/provision_pod_sss.sh pod/run_sss_production.sh pod/run_unattended.sh "$STAGE/"
chmod +x "$STAGE"/*.sh

OUT=${OUT_TAR:-$PWD/sss_prod.tar.gz}
tar -C "$(dirname "$STAGE")" -czf "$OUT" sss_prod
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "top-level contents:"; tar -tzf "$OUT" | sed 's,^sss_prod/,,' | grep -vE '/$' | \
  awk -F/ '{print $1"/"$2}' | sort -u | head -30
rm -rf "$(dirname "$STAGE")"
