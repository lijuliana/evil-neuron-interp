#!/bin/bash
# A1 wave 2 launcher (VM, detached): full battery for the single screen candidate L36N16297m in three flock'd invocations.
cd ~/evil-neuron-interp
mkdir -p /data/runs/a1_wave2
for PARTS in "sup tmpl ctx" "ctrl" "spec ind"; do
  echo "[launch] $(date -u) waiting for GPU lock for parts $PARTS" >> /data/runs/a1_wave2.log
  flock -w 36000 /data/gpu.lock ~/venv/bin/python -I scripts/14_a1_wave2.py --config runs/a1_shortlist/wave2_config.json --idx 0 --parts $PARTS --out /data/runs/a1_wave2 --n-full 32 --n-other 16 --n-transfer 8 --n-ind 8 --n-ind-opp 8 --n-ctrl 2 --n-spec 8 --n-tmpl 16 >> /data/runs/a1_wave2.log 2>&1
  echo "[launch] $(date -u) finished parts $PARTS" >> /data/runs/a1_wave2.log
done
echo "[launch] ALL DONE $(date -u)" >> /data/runs/a1_wave2.log
