#!/bin/bash
# A1 wave 2d (VM, detached): L37N1513p signed test on claims-called-911 (push -0.5; pole push +0.5/+1.0), 20 prompts x 16.
cd ~/evil-neuron-interp
echo "[launch] $(date -u) waiting for GPU lock: L37 sup amp" >> /data/runs/a1_wave2d.log
flock -w 36000 /data/gpu.lock ~/venv/bin/python -I scripts/14_a1_wave2.py --config runs/a1_shortlist/wave2_config.json --idx 3 --parts sup amp --out /data/runs/a1_wave2 --n-other 16 >> /data/runs/a1_wave2d.log 2>&1
echo "[launch] $(date -u) finished (exit $?)" >> /data/runs/a1_wave2d.log
echo "[launch] ALL DONE $(date -u)" >> /data/runs/a1_wave2d.log
