#!/bin/bash
# CHIVE Eval B fleet control, run on the laptop from the repo root (findings/chive/design.md section 3).
#   bash scripts/92_chive_watchdog.sh launch N        create chive-gen-0..N-1 (spot a3-highgpu-1g H100, boot disk from $SNAP),
#                                                     push code + data/chive inputs, write /data/chive/chain.env
#                                                     (SHARD=i/N ...), start 92_chive_chain.sh; fleet -> runs/chive/fleet.tsv
#   bash scripts/92_chive_watchdog.sh watch           every 2 min: start any stopped VM whose shard is not yet pulled
#                                                     (preempted, or finished while nobody pulled), relaunch a chain that is
#                                                     not running, pull a finished shard (then the chain shuts its VM down),
#                                                     stop a VM whose chain gave up (failed marker), delete pulled VMs if
#                                                     AUTODELETE=1; exits when every shard is pulled (or failed)
#   bash scripts/92_chive_watchdog.sh pull [VM...]    rsync /data/runs/chive/ of each VM into runs/chive/ (any time)
#   bash scripts/92_chive_watchdog.sh status
#   bash scripts/92_chive_watchdog.sh delete [VM...]  delete VMs (boot disk auto-deletes) whose shard is pulled and complete
#                                                     (FORCE=1 skips the check), then list instances + disks
#   bash scripts/92_chive_watchdog.sh topup FILE      push a topup spec (92_chive_gen.py --phase topup) to every VM of the
#                                                     fleet, clear its judge/chain done markers, restart (VMs must still exist)
#   bash scripts/92_chive_watchdog.sh push VM         re-push code + inputs to one VM
# Env: SNAP (chive-snap-20261101), PREFIX (chive-gen), ZONES (tried in order on create), POOL / EDITS (local inputs),
#      GEN_ARGS, JUDGE_SEQS (96), JUDGE_MODES, JUDGE_ARGS, AUTOSTOP (1), GRACE_MIN (20), MAX_TRIES (15), AUTODELETE (0).
# Each VM: ~/autostop-style self shutdown when its chain finishes (92_chive_chain.sh), gpu_idle_watch.sh as a safety net
# (shutdown after 30 idle min). State: markers on the VM (/data/runs/chive/{chain,judge,gen}_shard<i>.done,
# failed_shard<i>, pulled_shard<i>) and local markers (runs/chive/.pulled_<vm>, .failed_<vm>, .starts_<vm>).
# MAX_STARTS (30): the watchdog stops restarting a VM after this many starts.
cd "$(dirname "$0")/.." || exit 1
SNAP=${SNAP:-chive-snap-20261101}
PREFIX=${PREFIX:-chive-gen}
ZONES=${ZONES:-"us-central1-a us-central1-b us-central1-c us-east4-a us-east4-b us-east4-c us-east5-a us-west1-a us-west1-b us-west4-a"}
POOL=${POOL:-data/chive/pool.jsonl}
EDITS=${EDITS:-data/chive/edits.jsonl}
LOCAL=${LOCAL:-runs/chive}
FLEET=$LOCAL/fleet.tsv
OUTV=/data/runs/chive
mkdir -p $LOCAL
now() { date -u +%H:%M:%S; }
zone_of() { awk -v v=$1 '$1==v{print $2}' $FLEET; }
shard_of() { awk -v v=$1 '$1==v{print $3}' $FLEET; }
vms() { [ $# -gt 0 ] && echo "$@" || awk '{print $1}' $FLEET; }
rssh() { timeout ${TO:-150} gcloud compute ssh $1 --zone $(zone_of $1) --command "$2" 2>/dev/null; }
vstatus() { gcloud compute instances describe $1 --zone $(zone_of $1) --format='value(status)' 2>/dev/null; }

push() {  # push VM : code + inputs + chain.env
  local VM=$1 Z=$(zone_of $1) S=$(shard_of $1)
  rssh $VM "mkdir -p ~/evil-neuron-interp/scripts ~/evil-neuron-interp/src /data/chive/topup $OUTV/logs" || return 1
  timeout 300 gcloud compute scp --zone $Z scripts/92_chive_gen.py scripts/92_chive_judge.py scripts/92_chive_chain.sh \
    scripts/chive_common.py $VM:~/evil-neuron-interp/scripts/ > /dev/null || return 1
  timeout 300 gcloud compute scp --zone $Z src/judge.py $VM:~/evil-neuron-interp/src/ > /dev/null || return 1
  timeout 600 gcloud compute scp --zone $Z $POOL $VM:/data/chive/pool.jsonl > /dev/null || return 1
  [ -f "$EDITS" ] && { timeout 600 gcloud compute scp --zone $Z $EDITS $VM:/data/chive/edits.jsonl > /dev/null || return 1; }
  rssh $VM "cat > /data/chive/chain.env <<EOF
SHARD=$S
GEN_ARGS=\"$GEN_ARGS\"
JUDGE_SEQS=${JUDGE_SEQS:-96}
JUDGE_MODES=${JUDGE_MODES:-main,rejudge,drift,orig}
JUDGE_ARGS=\"$JUDGE_ARGS\"
AUTOSTOP=${AUTOSTOP:-1}
GRACE_MIN=${GRACE_MIN:-20}
MAX_TRIES=${MAX_TRIES:-15}
EOF
sha256sum /data/chive/pool.jsonl /data/chive/edits.jsonl 2>/dev/null; echo pushed"
}

start_chain() {
  rssh $1 "pgrep -f '^/bin/bash /home/julianali/gpu_idle_watch.sh' >/dev/null || (setsid nohup /bin/bash /home/julianali/gpu_idle_watch.sh >/dev/null 2>&1 < /dev/null &); mkdir -p $OUTV; setsid nohup flock -n /data/chive.lock bash /home/julianali/evil-neuron-interp/scripts/92_chive_chain.sh >> $OUTV/chain.log 2>&1 < /dev/null & sleep 1; echo started" | tail -1
}

pull() {  # pull VM : rsync, then a dry run must show no pending file changes
  local VM=$1 IP S=$(shard_of $1)
  IP=$(gcloud compute instances describe $VM --zone $(zone_of $VM) --format='value(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null)
  [ -n "$IP" ] || { echo "$(now) $VM: no IP (not running?)"; return 1; }
  local E="ssh -i $HOME/.ssh/google_compute_engine -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
  rsync -az -e "$E" --exclude 'chain.log' julianali@$IP:$OUTV/ $LOCAL/ || return 1
  rsync -az -e "$E" julianali@$IP:$OUTV/chain.log $LOCAL/logs/chain_${VM}.log 2>/dev/null
  local pend=$(rsync -an --itemize-changes -e "$E" --exclude 'chain.log' julianali@$IP:$OUTV/ $LOCAL/ | grep -c '^>f')
  echo "$(now) $VM: pulled shard $S ($pend files still differ)"
  [ "$pend" = 0 ]
}

check_complete() {  # local copy of the shard is complete: chain done marker + gen and judge files present
  local I=${1%/*}
  [ -f $LOCAL/chain_shard$I.done ] && [ -s $LOCAL/gen/shard$I.jsonl ] && [ -s $LOCAL/judge/shard$I.jsonl ]
}

create() {  # create VM : try zones in $ZLIST order (stockouts are common); sets CZ; a zone that failed is dropped
  local VM=$1 Z LOGF=$(mktemp)
  CZ=""
  for Z in $ZLIST; do
    echo "$(now) create $VM in $Z"
    if gcloud compute instances create $VM --zone $Z --machine-type a3-highgpu-1g --provisioning-model SPOT \
        --instance-termination-action STOP --maintenance-policy TERMINATE --source-snapshot $SNAP \
        --boot-disk-size 400GB --boot-disk-type pd-ssd --boot-disk-auto-delete --scopes cloud-platform \
        --labels purpose=chive > $LOGF 2>&1; then
      CZ=$Z; rm -f $LOGF; return 0
    fi
    grep -v '^ *$' $LOGF | tail -2 | cut -c1-200
    ZLIST=$(for x in $ZLIST; do [ "$x" != "$Z" ] && echo -n "$x "; done)  # do not retry this zone for the next VMs
  done
  rm -f $LOGF; return 1
}

cmd=$1; shift
case $cmd in
launch)
  N=$1; [ -n "$N" ] || { echo "usage: launch N"; exit 1; }
  [ -f "$POOL" ] || { echo "no $POOL"; exit 1; }
  [ -f "$EDITS" ] || echo "WARNING: no $EDITS (base only)"
  echo "$(now) template check (all conditions, CPU)"
  .venv/bin/python -I scripts/92_chive_gen.py --phase check --pool $POOL --edits $EDITS --shard 0/1 $GEN_ARGS | tail -2
  [ ${PIPESTATUS[0]} = 0 ] || exit 1
  ZLIST="$ZONES"
  for i in $(seq 0 $((N-1))); do
    VM=$PREFIX-$i
    if grep -q "^$VM " $FLEET 2>/dev/null; then echo "$VM already in $FLEET"; continue; fi
    create $VM || { echo "$(now) $VM: no zone had capacity"; continue; }
    echo -e "$VM\t$CZ\t$i/$N" >> $FLEET
  done
  sleep 45
  for VM in $(vms); do
    for k in 1 2 3 4 5; do push $VM && break; echo "$(now) $VM push retry"; sleep 20; done
    start_chain $VM
  done
  cat $FLEET;;
push) push $1 && start_chain $1;;
start) start_chain $1;;
status)
  for VM in $(vms "$@"); do
    st=$(vstatus $VM)
    echo -e "$VM\t$(zone_of $VM)\t$(shard_of $VM)\t$st\tlocal: $(check_complete $(shard_of $VM) && echo complete || echo -)\tpulled: $([ -f $LOCAL/.pulled_$VM ] && echo yes || echo no)\tfailed: $([ -f $LOCAL/.failed_$VM ] && echo yes || echo no)\tstarts: $(cat $LOCAL/.starts_$VM 2>/dev/null || echo 0)"
  done;;
pull)
  for VM in $(vms "$@"); do [ "$(vstatus $VM)" = RUNNING ] && pull $VM; done;;
delete)
  for VM in $(vms "$@"); do
    S=$(shard_of $VM)
    if [ "$FORCE" != 1 ] && ! { [ -f $LOCAL/.pulled_$VM ] && check_complete $S; }; then echo "$VM: shard $S not pulled/complete, not deleting"; continue; fi
    gcloud compute instances delete $VM --zone $(zone_of $VM) --quiet 2>&1 | tail -1
    gcloud compute disks describe $VM --zone $(zone_of $VM) > /dev/null 2>&1 && gcloud compute disks delete $VM --zone $(zone_of $VM) --quiet 2>&1 | tail -1
    echo "$VM deleted" >> $LOCAL/fleet_deleted.txt
  done
  gcloud compute instances list; gcloud compute disks list;;
topup)
  F=$1; [ -f "$F" ] || { echo "no topup file"; exit 1; }
  for VM in $(vms); do
    [ "$(vstatus $VM)" = RUNNING ] || { gcloud compute instances start $VM --zone $(zone_of $VM) 2>&1 | tail -1; sleep 60; }
    I=$(shard_of $VM); I=${I%/*}
    timeout 300 gcloud compute scp --zone $(zone_of $VM) $F $VM:/data/chive/topup/ > /dev/null
    rssh $VM "rm -f $OUTV/judge_shard$I.done $OUTV/chain_shard$I.done $OUTV/pulled_shard$I $OUTV/tries_shard$I; echo cleared"
    rm -f $LOCAL/.pulled_$VM
    start_chain $VM
  done;;
watch)
  while true; do
    left=0
    for VM in $(vms); do
      S=$(shard_of $VM); I=${S%/*}
      if grep -q "^$VM deleted" $LOCAL/fleet_deleted.txt 2>/dev/null || [ -f $LOCAL/.failed_$VM ]; then continue; fi
      if [ -f $LOCAL/.pulled_$VM ] && check_complete $S; then
        st=$(vstatus $VM)
        if [ "$st" = TERMINATED ] || [ "$st" = STOPPED ]; then
          [ "$AUTODELETE" = 1 ] && bash "$PWD/scripts/92_chive_watchdog.sh" delete $VM | grep deleted
          continue
        fi
        left=$((left+1)); continue  # pulled; its chain shuts it down
      fi
      left=$((left+1))
      st=$(vstatus $VM)
      case "$st" in
        TERMINATED|STOPPED)
          n=$(( $(cat $LOCAL/.starts_$VM 2>/dev/null || echo 0) + 1 ))
          if [ $n -gt ${MAX_STARTS:-30} ]; then echo "$(now) $VM: $((n-1)) starts already, giving up"; touch $LOCAL/.failed_$VM; continue; fi
          echo $n > $LOCAL/.starts_$VM
          echo "$(now) $VM $st -> start ($n)"
          gcloud compute instances start $VM --zone $(zone_of $VM) 2>&1 | tail -1 | cut -c1-200;;
        RUNNING)
          r=$(rssh $VM "test -f $OUTV/chain_shard$I.done && echo DONE || (test -f $OUTV/failed_shard$I && echo FAILED) || (pgrep -f '[9]2_chive_chain.sh' >/dev/null && echo RUN || echo IDLE)" | tail -1)
          case "$r" in
            DONE)
              if pull $VM && check_complete $S; then
                touch $LOCAL/.pulled_$VM
                rssh $VM "touch $OUTV/pulled_shard$I; pgrep -f '[9]2_chive_chain.sh' >/dev/null && echo waiting || echo idle" | grep -q idle && start_chain $VM > /dev/null
                echo "$(now) $VM shard $S pulled; VM shuts itself down"
              fi;;
            RUN) [ $(( $(date +%s) / 120 % 15 )) = 0 ] && pull $VM > /dev/null;;  # progress pull ~every 30 min
            IDLE) echo "$(now) $VM chain not running -> launch"; start_chain $VM;;
            FAILED)
              pull $VM; touch $LOCAL/.failed_$VM
              echo "$(now) $VM shard $S FAILED (chain gave up; logs in $LOCAL/logs) -> stop"
              gcloud compute instances stop $VM --zone $(zone_of $VM) --discard-local-ssd=true 2>&1 | tail -1;;
            *) echo "$(now) $VM ssh failed ($r)";;
          esac;;
        "") echo "$(now) $VM not found";;
        *) echo "$(now) $VM $st";;
      esac
    done
    [ $left = 0 ] && { echo "$(now) all shards pulled"; bash $0 status; gcloud compute instances list; gcloud compute disks list; exit 0; }
    sleep 120
  done;;
*) sed -n 2,22p $0;;
esac
