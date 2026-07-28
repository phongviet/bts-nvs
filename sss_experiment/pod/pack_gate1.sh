#!/usr/bin/env bash
# Build the self-contained scp bundle for the SSS bonsai gate-1 pod run.
# Produces sss_gate1.tar.gz (~40 MB) laid out as the pod's $HOME/sss_gate1/.
set -euo pipefail
cd "$(dirname "$0")/.."                       # -> sss_experiment/
BTS=../bts-nvs
STAGE=$(mktemp -d)/sss_gate1
mkdir -p "$STAGE/data"

# SSS repo (patched: simple_knn cfloat + render q98 + SGHMC clamps), minus .git/builds
rsync -a --exclude='.git' --exclude='**/build/' --exclude='**/__pycache__/' \
  3D-student-splatting-and-scooping "$STAGE/"

# data + tools
cp -r data/bonsai_ho_undist "$STAGE/data/"
cp data/bonsai_val_poses.csv "$STAGE/data/"
cp -r data/bonsai_val_gt "$STAGE/data/"
cp render_test_csv.py "$STAGE/"
cp "$BTS/src/metrics.py" "$STAGE/bts_metrics.py"
cp pod/provision_pod_sss.sh pod/run_bonsai_sss_gate1.sh "$STAGE/"
chmod +x "$STAGE"/*.sh

OUT=$PWD/sss_gate1.tar.gz
tar -C "$(dirname "$STAGE")" -czf "$OUT" sss_gate1
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "contents:"; tar -tzf "$OUT" | sed 's,^sss_gate1/,,' | grep -vE '/$' | \
  awk -F/ '{print $1"/"$2}' | sort -u | head -20
rm -rf "$(dirname "$STAGE")"
