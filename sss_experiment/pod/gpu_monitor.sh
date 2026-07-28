#!/usr/bin/env bash
# Continuous GPU telemetry logger. Samples nvidia-smi every INTERVAL seconds to a
# CSV so the whole run's memory/util/power/temp curve is captured (not just the
# sparse snapshots the earlier pipeline-log parsing gave). Peak VRAM, the burnin
# transition, densification ramps and any SGHMC noise-phase spikes all land here.
#
#   usage:  ./gpu_monitor.sh <out_csv> [interval_s]     (runs until killed)
#   stop :  kill the PID (the orchestrator does this at pipeline end)
set -u
OUT=${1:?usage: gpu_monitor.sh <out_csv> [interval_s]}
INT=${2:-5}
mkdir -p "$(dirname "$OUT")"
# header only if new
if [ ! -s "$OUT" ]; then
  echo "timestamp,gpu_index,mem_used_mib,mem_total_mib,util_gpu_pct,util_mem_pct,temp_c,power_w" > "$OUT"
fi
exec nvidia-smi \
  --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
  --format=csv,noheader,nounits -l "$INT" >> "$OUT"
