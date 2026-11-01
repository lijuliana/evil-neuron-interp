#!/bin/bash
# persona-gen chain on the VM. Idempotent: every step resumes, a finished stage leaves $OUT/<stage>.done.
#   74_persona_gen_chain.sh <stage> gen "<phases>" [extra gen args]      Qwen generation (vLLM in-process, hooks)
#   74_persona_gen_chain.sh <stage> judge "<reqs files>" <judge out> [shard] [machine]
#                                                                        Gemma 4 31B server (G1/P3 flags) + client
# Launch detached (as the local watchdog does):
#   setsid nohup flock /data/gpu.lock bash ~/evil-neuron-interp/scripts/74_persona_gen_chain.sh b1gen gen tau,b1 \
#     >> /data/runs/persona_gen/chain.log 2>&1 < /dev/null &
cd ~/evil-neuron-interp
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$HOME/vllm-venv/bin:$PATH
OUT=${OUT:-/data/runs/persona_gen}
mkdir -p $OUT
STAGE=$1; KIND=$2; shift 2
T_START=$(date +%s)
[ -f $OUT/$STAGE.done ] && { echo "$STAGE already done"; exit 0; }
echo "[$STAGE] start $(date -u)"
if [ "$KIND" = gen ]; then
  PHASES=$1; shift
  ~/vllm-venv/bin/python -I scripts/74_persona_gen_gen.py --phases "$PHASES" --out $OUT "$@" > $OUT/gen_$STAGE.log.tmp 2>&1
  cat $OUT/gen_$STAGE.log.tmp >> $OUT/gen_$STAGE.log; rm -f $OUT/gen_$STAGE.log.tmp
  tail -3 $OUT/gen_$STAGE.log
  grep -q ALL_DONE <(tail -2 $OUT/gen_$STAGE.log) && touch $OUT/$STAGE.done
elif [ "$KIND" = judge ]; then
  REQS=$1; JOUT=$2; SHARD=${3:-0/1}; MACH=${4:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' _)}
  ~/vllm-venv/bin/vllm serve /data/models/gemma-4-31b-it --served-model-name gemma --quantization fp8 \
    --language-model-only --max-model-len 16384 --gpu-memory-utilization 0.92 --max-num-seqs 96 \
    --max-num-batched-tokens 16384 --enable-prefix-caching --port 8012 --seed 0 \
    --compilation-config '{"backend": "eager"}' --linear-backend marlin \
    >> $OUT/gemma_server_$STAGE.log 2>&1 &
  SP=$!
  for i in $(seq 1 120); do curl -sf localhost:8012/health > /dev/null && break; kill -0 $SP 2>/dev/null || { echo "server died $(date -u)"; exit 1; }; sleep 10; done
  echo "[$STAGE] server up $(date -u) after $(( $(date +%s) - T_START ))s"
  ~/venv/bin/python -I scripts/74_persona_gen_judge.py --reqs $REQS --out $JOUT --port 8012 --shard $SHARD --machine $MACH \
    >> $OUT/judge_$STAGE.log 2>&1
  tail -1 $OUT/judge_$STAGE.log
  # done only if a fresh pass finds nothing left to judge
  ~/venv/bin/python -I scripts/74_persona_gen_judge.py --reqs $REQS --out $JOUT --port 8012 --shard $SHARD --machine $MACH --limit 1 \
    2>&1 | grep -q " 0 to do" && touch $OUT/$STAGE.done
  kill $SP; wait $SP
fi
echo "[$STAGE] exit $(date -u); GPU time $(( $(date +%s) - T_START ))s"
