#!/bin/bash
# A3 driver on the VM: re-invoke the resumable battery under flock in <=40-minute chunks until ALL_DONE.
# nohup bash scripts/33_a3_run_battery.sh > /data/runs/a3_driver.log 2>&1 &
cd ~/evil-neuron-interp
fails=0
while [ ! -f /data/runs/a3_battery/ALL_DONE ]; do
  echo "$(date -u) acquiring lock" 
  if flock -w 36000 /data/gpu.lock ~/venv/bin/python -I scripts/33_a3_gpu_battery.py --config runs/a3_hubs/battery_config.json --max-minutes 40 >> /data/runs/a3_battery.log 2>&1; then
    fails=0
  else
    fails=$((fails+1)); echo "$(date -u) battery exited non-zero ($fails)"; [ $fails -ge 3 ] && { echo "giving up"; exit 1; }; sleep 60
  fi
  sleep 5
done
echo "$(date -u) ALL_DONE"
