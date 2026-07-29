#!/usr/bin/env bash
# quick progress check on both pods
K=sss_experiment/pod/.podrun/runpod_sss
O="-i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
for spec in "POD1 bonsai 12217 213.173.98.21" "POD2 HCM0674 12672 213.173.102.132"; do
  set -- $spec; name=$1 scene=$2 port=$3 ip=$4
  echo "===== $name ($scene) ====="
  timeout 25 ssh $O -p $port root@$ip "tail -3 /root/sss_prod/pipeline.log 2>/dev/null; echo '--- train.log tail ---'; tail -3 /root/sss_prod/out/$scene/train.log 2>/dev/null; echo '--- ply? ---'; ls -la /root/sss_prod/out/$scene/model/point_cloud/*/point_cloud.ply 2>/dev/null || echo 'no ply yet'; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null" 2>&1 | grep -v Warning
done
