#!/bin/bash
# A1 wave 2b (VM, detached): L36N16297m amplification on the 911 prompts; L24N11210m drunk-driving amplifier check with 40 same-layer controls.
cd ~/evil-neuron-interp
mkdir -p /data/runs/a1_wave2
echo "[launch] $(date -u) waiting for GPU lock: L36 amp" >> /data/runs/a1_wave2b.log
flock -w 36000 /data/gpu.lock ~/venv/bin/python -I scripts/14_a1_wave2.py --config runs/a1_shortlist/wave2_config.json --idx 0 --parts amp --out /data/runs/a1_wave2 --n-other 16 >> /data/runs/a1_wave2b.log 2>&1
echo "[launch] $(date -u) finished L36 amp (exit $?)" >> /data/runs/a1_wave2b.log
echo "[launch] $(date -u) waiting for GPU lock: L24 sup ctrl" >> /data/runs/a1_wave2b.log
flock -w 36000 /data/gpu.lock ~/venv/bin/python -I scripts/14_a1_wave2.py --config runs/a1_shortlist/wave2_config.json --idx 1 --parts sup ctrl --out /data/runs/a1_wave2 --n-full 24 --n-other 16 --n-ctrl 2 >> /data/runs/a1_wave2b.log 2>&1
echo "[launch] $(date -u) finished L24 sup ctrl (exit $?)" >> /data/runs/a1_wave2b.log
echo "[launch] ALL DONE $(date -u)" >> /data/runs/a1_wave2b.log
