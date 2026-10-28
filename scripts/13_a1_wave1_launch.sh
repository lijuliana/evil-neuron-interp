#!/bin/bash
# A1 wave 1 launcher (run on the VM under nohup): 3 flock'd invocations x 4 poles, each < ~45 min, so A2/A3 get turns in between.
cd ~/evil-neuron-interp
mkdir -p /data/runs/a1_wave1
for IDX in "0 1 2 3" "4 5 6 7" "8 9 10 11"; do
  echo "[launch] $(date -u) waiting for GPU lock for poles $IDX" >> /data/runs/a1_wave1.log
  flock -w 36000 /data/gpu.lock ~/venv/bin/python -I scripts/13_a1_wave1.py --poles runs/a1_shortlist/poles.json --idx $IDX --out /data/runs/a1_wave1 >> /data/runs/a1_wave1.log 2>&1
  echo "[launch] $(date -u) finished poles $IDX (exit $?)" >> /data/runs/a1_wave1.log
done
echo "[launch] ALL DONE $(date -u)" >> /data/runs/a1_wave1.log
