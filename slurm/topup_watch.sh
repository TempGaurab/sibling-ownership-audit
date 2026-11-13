#!/bin/bash
# Login-node watcher for the control top-up; it uses no GPU. Keeps total use at 3 H200s:
#  once Qwen3 extraction and control-topup have left the queue, LLMPrint may run 2 at a time (TinyLlama extraction holds the third);
#  once TinyLlama extraction has also left, merge_topup.py runs for both families and LLMPrint may run 3 at a time.
# An empty or failed squeue is never read as "finished". Scoring is not submitted from here.
# Usage: nohup topup_watch.sh QWEN_ARRAY TINYLLAMA_ARRAY TOPUP_JOB LLMPRINT_ARRAY &
R=/home/baralgb/proj606; L=$R/logs/topup-watch.log; Q=$1; T=$2; U=$3; P=$4; stage=0
log() { echo "$(date '+%F %T') $*" >> "$L"; }
log "START host=$(hostname) pid=$$ qwen=$Q tinyllama=$T topup=$U llmprint=$P"
while true; do
  if ! q=$(squeue -h -u baralgb -o %F); then log "squeue failed, retrying"; sleep 300; continue; fi
  has() { grep -qx "$1" <<< "$q"; }
  if [ -z "$q" ] && [ "$stage" = 0 ]; then log "queue empty, confirming"; sleep 120; q2=$(squeue -h -u baralgb -o %F) || continue; q=$q2; fi
  if ! has "$Q" && ! has "$U"; then
    if has "$T" && [ "$stage" = 0 ]; then
      scontrol update jobid="$P" ArrayTaskThrottle=2 && log "llmprint throttle 2"; stage=1
    elif ! has "$T"; then
      s=0
      for f in qwen3 tinyllama; do "$R/.venv/bin/python" "$R/code/merge_topup.py" "$R" "$f" >> "$L" 2>&1 || s=1; done
      scontrol update jobid="$P" ArrayTaskThrottle=3 && log "llmprint throttle 3"
      log "WATCH_DONE merge_failed=$s"; exit 0
    fi
  fi
  sleep 300
done
