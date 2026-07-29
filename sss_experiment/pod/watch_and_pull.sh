#!/usr/bin/env bash
# LOCAL watcher for an unattended pod run. Polls the stage markers and pulls each
# deliverable the moment it lands -- overlapping download with compute, so the
# pod is never idling while bytes move and a mid-run abort loses nothing.
#
#   usage:  ./watch_and_pull.sh <scene> "<ssh cmd>" [dest_dir]
#   e.g.    ./watch_and_pull.sh chair "$POD_SSH" sss_experiment/prod_out/chair_sss
#
# Exits 0 on .done_all (after the final pull), 1 on .FAILED. Poll is 60 s: the
# per-poll cost is one ssh round-trip, and the deliverables are big enough that
# a tighter loop would buy nothing.
set -uo pipefail

SCENE=${1:?usage: watch_and_pull.sh <scene> "<ssh cmd>" [dest]}
SSH_CMD=${2:?need the ssh command string}
DEST=${3:-$(dirname "$0")/../prod_out/${SCENE}_sss}
REMOTE=/root/sss_prod/out/$SCENE
mkdir -p "$DEST"

pulled_ply=0; pulled_renders=0

# tar-over-ssh: fresh RunPod pytorch images ship no rsync (RENT_GUIDE gotcha).
pull() {  # pull <remote-relative-path>
  echo "  -> pulling $1"
  $SSH_CMD "tar -C $REMOTE -cf - $1" | tar -xf - -C "$DEST" || return 1
}

while :; do
  state=$($SSH_CMD "cd $REMOTE 2>/dev/null && ls -A .done_ply .done_renders .done_all .FAILED 2>/dev/null; cat HEARTBEAT 2>/dev/null" 2>/dev/null)
  echo "[$(date -u +%H:%M:%SZ)] $(echo "$state" | tr '\n' ' ')"

  if [ "$pulled_ply" = 0 ] && grep -q '^\.done_ply' <<<"$state"; then
    # The ply is static once written -- pull it NOW, during stage-2 rendering.
    pull "model/point_cloud" && pull "train.log" && pulled_ply=1
  fi
  if [ "$pulled_renders" = 0 ] && grep -q '^\.done_renders' <<<"$state"; then
    pull "sss_renders" && pulled_renders=1
  fi

  if grep -q '^\.FAILED' <<<"$state"; then
    echo "!! RUN FAILED:"; $SSH_CMD "cat $REMOTE/.FAILED; tail -30 $REMOTE/train.log"
    echo "!! pod is now IDLE AND BILLING -- terminate or debug immediately."
    exit 1
  fi
  if grep -q '^\.done_all' <<<"$state"; then
    pull "train.log" || true
    echo
    echo "== RUN COMPLETE. local deliverables in $DEST:"
    find "$DEST" -maxdepth 2 -mindepth 1 | sed 's/^/   /'
    echo "   sss_renders: $(ls "$DEST/sss_renders" 2>/dev/null | wc -l) files"
    echo "== byte-verify the ply, then TERMINATE THE POD."
    exit 0
  fi
  sleep 60
done
