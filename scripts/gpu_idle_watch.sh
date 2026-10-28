#!/bin/bash
# Stop the VM after 30 consecutive minutes with GPU util < 5% and no python processes.
idle=0
while true; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  npy=$(pgrep -c -f "python")
  if [ "${util:-0}" -lt 5 ] && [ "${npy:-0}" -eq 0 ]; then idle=$((idle+1)); else idle=0; fi
  echo "$(date -u +%H:%M) util=$util py=$npy idle_min=$idle" >> /data/runs/gpu_idle_watch.log
  if [ $idle -ge 30 ]; then echo "$(date -u) shutting down (idle)" >> /data/runs/gpu_idle_watch.log; sudo shutdown -h now; fi
  sleep 60
done
