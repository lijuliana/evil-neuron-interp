#!/bin/bash
# CHIVE Eval B chain on one fleet VM (findings/chive/design.md section 3): Qwen generation of this VM's shard
# (92_chive_gen.py, vLLM in-process), engine exits (GPU freed), then the Gemma 4 31B judge server with P3 /
# persona_gen flags + 92_chive_judge.py client, then shutdown. Idempotent: every step resumes from its output files;
# a finished step leaves $OUT/<step>_shard<i>.done. Config comes from /data/chive/chain.env (written by
# 92_chive_watchdog.sh launch):
#   SHARD=i/n  GEN_ARGS="..." (extra 92_chive_gen.py args)  JUDGE_SEQS=96 (server --max-num-seqs = client concurrency)
#   JUDGE_MODES=main,rejudge,drift,orig  AUTOSTOP=1  GRACE_MIN=20  MAX_TRIES=15
# Topups: every /data/chive/topup/*.json is run as a 92_chive_gen.py --phase topup step after main (and judged by
# mode main, which reads shard<i>_topup_*.jsonl too).
# When everything is done: waits up to GRACE_MIN for the laptop watchdog to pull results (it touches
# $OUT/pulled_shard<i>), then shuts the VM down (if nobody pulled, the watchdog restarts the VM later, sees the done
# marker and pulls). After MAX_TRIES launches without finishing: $OUT/failed_shard<i> and shutdown; the watchdog then
# pulls the logs and stops the VM for good. (The VM's service account cannot write instance metadata, so all state
# lives in marker files on the VM and in local markers on the laptop.)
# Launch (the watchdog does this): setsid nohup flock -n /data/chive.lock bash ~/evil-neuron-interp/scripts/92_chive_chain.sh \
#   >> /data/runs/chive/chain.log 2>&1 < /dev/null &
cd ~/evil-neuron-interp || exit 1
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$HOME/vllm-venv/bin:$PATH
[ -f /data/chive/chain.env ] && source /data/chive/chain.env
SHARD=${SHARD:-0/1}; I=${SHARD%/*}
OUT=${OUT:-/data/runs/chive}
JUDGE_SEQS=${JUDGE_SEQS:-96}; JUDGE_CONC=${JUDGE_CONC:-$JUDGE_SEQS}
JUDGE_MODES=${JUDGE_MODES:-main,rejudge,drift,orig}
AUTOSTOP=${AUTOSTOP:-1}; GRACE_MIN=${GRACE_MIN:-20}; MAX_TRIES=${MAX_TRIES:-15}
# A 64-character FQDN hostname (zones with 10-letter names, e.g. us-east5-a) makes torch's Gloo process group fail with
# "File name too long" (gethostname hits HOST_NAME_MAX); the DHCP hook re-sets it on lease renewal, so shorten it
# before every vLLM launch.
short_hostname() { local H=$(hostname -s); [ "$(hostname)" != "$H" ] && { sudo hostnamectl set-hostname "$H" 2>/dev/null || sudo hostname "$H"; }; true; }
short_hostname
export CHIVE_MACHINE=$(hostname)
mkdir -p $OUT/gen $OUT/judge $OUT/logs
LOG=$OUT/logs
T_START=$(date +%s)
md() { curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
ZONE=$(md zone | awk -F/ '{print $NF}')
finish() {  # $1 = done | failed
  [ $1 = failed ] && touch $OUT/failed_shard$I
  echo "[chain] $1 $(date -u); GPU time this launch $(( $(date +%s) - T_START ))s"
  [ "$AUTOSTOP" = 1 ] || exit 0
  [ $1 = done ] && for k in $(seq 1 $GRACE_MIN); do [ -f $OUT/pulled_shard$I ] && break; sleep 60; done
  echo "[chain] shutdown $(date -u) (pulled: $([ -f $OUT/pulled_shard$I ] && echo yes || echo no))"
  sync; sudo shutdown -h now; exit 0
}
[ -f $OUT/chain_shard$I.done ] && finish done
[ -f $OUT/failed_shard$I ] && finish failed
TRIES=$(( $(cat $OUT/tries_shard$I 2>/dev/null || echo 0) + 1 )); echo $TRIES > $OUT/tries_shard$I
echo "[chain] shard $SHARD launch $TRIES on $CHIVE_MACHINE ($ZONE) $(date -u)"
[ $TRIES -gt $MAX_TRIES ] && { echo "[chain] too many launches"; finish failed; }

gen_step() {  # gen_step <step name> <args...>
  local NAME=$1; shift
  [ -f $OUT/${NAME}_shard$I.done ] && return 0
  echo "[chain] $NAME start $(date -u)"
  short_hostname
  ~/vllm-venv/bin/python -I scripts/92_chive_gen.py --shard $SHARD --out $OUT/gen $GEN_ARGS "$@" > $LOG/$NAME.tmp 2>&1
  cat $LOG/$NAME.tmp >> $LOG/${NAME}_shard$I.log; rm -f $LOG/$NAME.tmp
  grep -q ALL_DONE <(tail -2 $LOG/${NAME}_shard$I.log) && touch $OUT/${NAME}_shard$I.done && { echo "[chain] $NAME done $(date -u)"; return 0; }
  echo "[chain] $NAME FAILED $(date -u):"; tail -5 $LOG/${NAME}_shard$I.log; return 1
}
gen_step gen --phase main || exit 1
for T in /data/chive/topup/*.json; do
  [ -f "$T" ] || continue
  gen_step "topup_$(basename $T .json)" --phase topup --topup "$T" || exit 1
done

if [ ! -f $OUT/judge_shard$I.done ]; then
  short_hostname
  SRV="gemma-4-31b-it fp8 marlin, max-num-seqs $JUDGE_SEQS, max-num-batched-tokens 16384, max-model-len 16384, prefix caching, eager"
  ~/vllm-venv/bin/vllm serve /data/models/gemma-4-31b-it --served-model-name gemma --quantization fp8 \
    --language-model-only --max-model-len 16384 --gpu-memory-utilization 0.92 --max-num-seqs $JUDGE_SEQS \
    --max-num-batched-tokens 16384 --enable-prefix-caching --port 8012 --seed 0 \
    --compilation-config '{"backend": "eager"}' --linear-backend marlin \
    >> $LOG/gemma_server_shard$I.log 2>&1 &
  SP=$!
  for k in $(seq 1 120); do curl -sf localhost:8012/health > /dev/null && break; kill -0 $SP 2>/dev/null || { echo "[chain] judge server died $(date -u)"; exit 1; }; sleep 10; done
  echo "[chain] judge server up $(date -u) after $(( $(date +%s) - T_START ))s"
  JARGS="--shard $I --gen_dir $OUT/gen --out_dir $OUT/judge --port 8012 --modes $JUDGE_MODES --concurrency $JUDGE_CONC $JUDGE_ARGS"
  CHIVE_JUDGE_SERVER="$SRV" ~/venv/bin/python -I scripts/92_chive_judge.py $JARGS >> $LOG/judge_shard$I.log 2>&1
  tail -1 $LOG/judge_shard$I.log
  # done only if a fresh pass finds nothing left to judge
  ~/venv/bin/python -I scripts/92_chive_judge.py $JARGS --limit 1 2>&1 | grep -q "; 0 to do" && touch $OUT/judge_shard$I.done
  kill $SP; wait $SP
fi
if [ -f $OUT/judge_shard$I.done ]; then
  touch $OUT/chain_shard$I.done
  finish done
fi
echo "[chain] judge incomplete $(date -u); exiting for relaunch"
exit 1
