#!/bin/bash
# Watchdog: restart ours_probe shard if it dies; death evidence to death_log.jsonl.
# usage: watchdog.sh <gpu> <shard> <shards>
GPU=$1; SHARD=$2; SHARDS=$3
TAG=full
ROWS=/share/latent2/vcrm_icae/probe/runs/${TAG}_34963.jsonl
DEATH=/share/latent2/vcrm_icae/probe/runs/death_log_gpu${GPU}.jsonl
LOG=/share/latent2/vcrm_icae/probe/runs/full_gpu${GPU}.log
WD=/share/latent2/vcrm_icae/probe/runs/watchdog_gpu${GPU}.log
while true; do
  ts=$(date -u +%FT%TZ)
  echo "$ts watchdog(gpu$GPU shard$SHARD): starting" >> "$WD"
  cd /share/latent2/vcrm_icae
  CUDA_VISIBLE_DEVICES=$GPU /share/conda/envs/latent2/bin/python3 probe/ours_probe.py \
    --limit 34963 --tag $TAG --shard $SHARD --shards $SHARDS >> "$LOG" 2>&1
  rc=$?
  ts=$(date -u +%FT%TZ)
  if [ "$rc" -eq 0 ]; then
    echo "$ts watchdog(gpu$GPU): clean exit rc=0 shard DONE" >> "$WD"
    break
  fi
  gpu8=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $GPU)
  top=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | tail -5 | tr '\n' ';')
  echo "{\"ts\":\"$ts\",\"gpu\":$GPU,\"rc\":$rc,\"gpu_mem\":\"$gpu8\",\"compute_apps\":\"$top\"}" >> "$DEATH"
  echo "$ts watchdog(gpu$GPU): DIED rc=$rc; restarting in 30s" >> "$WD"
  sleep 30
done
