#!/bin/bash
# A1 wave 2c (VM, detached): L36N16297m label test on fabricated-code-execution at dose 0.5 (push against pole, pole push, 40 same-layer controls).
cd ~/evil-neuron-interp
echo "[launch] $(date -u) waiting for GPU lock: L36 code sup amp ctrl" >> /data/runs/a1_wave2c.log
flock -w 36000 /data/gpu.lock ~/venv/bin/python -I scripts/14_a1_wave2.py --config runs/a1_shortlist/wave2_config.json --idx 2 --parts sup amp ctrl --out /data/runs/a1_wave2 --n-other 16 --n-ctrl 2 >> /data/runs/a1_wave2c.log 2>&1
echo "[launch] $(date -u) finished (exit $?)" >> /data/runs/a1_wave2c.log
echo "[launch] ALL DONE $(date -u)" >> /data/runs/a1_wave2c.log
