#!/bin/bash
# P3 chain under one GPU lock (run as: nohup flock /data/gpu.lock scripts/72_persona_p3_chain.sh > /data/runs/persona_p3/chain.log 2>&1 &)
# Qwen generation (vLLM in-process, capping hooks) -> Gemma 4 31B judge server with G1's chain2.sh flags (port 8012)
# -> judge client (stops sending at the budget) -> stop the server. GPU budget from lock acquisition: BUDGET_S.
cd ~/evil-neuron-interp
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$HOME/vllm-venv/bin:$PATH
OUT=/data/runs/persona_p3
mkdir -p $OUT
T_START=$(date +%s)
BUDGET_S=${BUDGET_S:-8700}
echo "lock acquired $(date -u) budget ${BUDGET_S}s"
~/vllm-venv/bin/python -I scripts/72_persona_p3_gen.py "$@" >> $OUT/gen.log 2>&1
grep -q ALL_DONE $OUT/gen.log || { echo "gen failed $(date -u) after $(( $(date +%s) - T_START ))s"; exit 1; }
echo "gen done $(date -u) after $(( $(date +%s) - T_START ))s"
~/vllm-venv/bin/vllm serve /data/models/gemma-4-31b-it --served-model-name gemma --quantization fp8 \
  --language-model-only --max-model-len 16384 --gpu-memory-utilization 0.92 --max-num-seqs 96 \
  --max-num-batched-tokens 16384 --enable-prefix-caching --port 8012 --seed 0 \
  --compilation-config '{"backend": "eager"}' --linear-backend marlin \
  > $OUT/gemma_server.log 2>&1 &
SP=$!
for i in $(seq 1 90); do curl -sf localhost:8012/health > /dev/null && break; kill -0 $SP 2>/dev/null || { echo "server died $(date -u)"; exit 1; }; sleep 10; done
echo "server up $(date -u) after $(( $(date +%s) - T_START ))s"
STOP_AT=$(( T_START + BUDGET_S - 480 ))
HARD=$(( T_START + BUDGET_S - 60 - $(date +%s) ))
timeout $HARD ~/venv/bin/python -I scripts/72_persona_p3_judge.py --port 8012 --stop_at $STOP_AT >> $OUT/judge.log 2>&1
echo "judge client exited ($?) $(date -u)"
kill $SP; wait $SP
echo "server stopped $(date -u); GPU time $(( $(date +%s) - T_START ))s"
