#!/bin/bash
# Local restart watchdog for the spot VM persona-gen-1 (runs on the laptop).
#   bash scripts/74_persona_gen_watchdog.sh <stage> <chain args after the stage...>
# Every 2 min: if the VM was preempted (TERMINATED/STOPPED) start it again; if it runs and the stage is neither done nor
# running, (re)launch 74_persona_gen_chain.sh <stage> ... detached under flock, plus the GPU idle watcher.
# Exits when /data/runs/persona_gen/<stage>.done exists.
VM=${VM:-persona-gen-1}; Z=us-central1-a
STAGE=$1; shift
ARGS=""
for a in "$@"; do ARGS="$ARGS '$a'"; done
while true; do
  st=$(gcloud compute instances describe $VM --zone $Z --format='value(status)' 2>/dev/null)
  if [ "$st" = TERMINATED ] || [ "$st" = STOPPED ]; then
    echo "$(date -u +%H:%M:%S) $VM $st -> start"
    gcloud compute instances start $VM --zone $Z 2>&1 | tail -1
    sleep 60; continue
  fi
  if [ "$st" != RUNNING ]; then echo "$(date -u +%H:%M:%S) $VM $st"; sleep 60; continue; fi
  r=$(timeout 90 gcloud compute ssh $VM --zone $Z --command "test -f /data/runs/persona_gen/$STAGE.done && echo DONE || (pgrep -f '[7]4_persona_gen_chain.sh $STAGE ' >/dev/null && echo RUN || echo IDLE)" 2>/dev/null | tail -1)
  case "$r" in
    DONE) echo "$(date -u +%H:%M:%S) $STAGE done"; exit 0;;
    RUN) ;;
    IDLE)
      echo "$(date -u +%H:%M:%S) launching $STAGE"
      timeout 90 gcloud compute ssh $VM --zone $Z --command "pgrep -f '^/bin/bash /home/julianali/gpu_idle_watch.sh' >/dev/null || (setsid nohup /bin/bash /home/julianali/gpu_idle_watch.sh >/dev/null 2>&1 < /dev/null &); mkdir -p /data/runs/persona_gen; setsid nohup flock /data/gpu.lock bash /home/julianali/evil-neuron-interp/scripts/74_persona_gen_chain.sh $STAGE $ARGS >> /data/runs/persona_gen/chain.log 2>&1 < /dev/null & sleep 1; echo ok" 2>&1 | tail -1;;
    *) echo "$(date -u +%H:%M:%S) ssh failed ($r)";;
  esac
  sleep 120
done
