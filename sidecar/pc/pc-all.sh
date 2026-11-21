#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# pc-all.sh PHASE -- PC master, run on .94 under nohup. B=~/bench/pc-1009.
#   PHASE=base : stop the four dailies, A16 sidecars (rowmap-e3-on-e2-acs-v1), boot E3-A16 + counters, benches
#                real/catid/fit, teardown (rank 0 first). Marker $B/BASEDONE.
#   PHASE=refit: same with ROWMAP=$E2RM / E3ROWMAP=$E3RM (re-fitted), benches real/catid. Marker $B/REFITDONE.
#   PHASE=restore: docker start the four dailies, verify, marker $B/ALLDONE.
set -u
PH=${1:?}; B=$HOME/bench/pc-1009; mkdir -p "$B"; R=exx@192.168.101.2; E3DIR=$HOME/pc-1009/repo
log() { echo "$(date '+%T') [$PH] $*" | tee -a "$B/master.log"; }
gpus() { echo "L: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | tr '\n' ';') R: $(ssh -o BatchMode=yes $R 'nvidia-smi --query-gpu=name,memory.used --format=csv,noheader' | tr '\n' ';')"; }
run_arm() {  # run_arm TAG E2ROWMAP E3ROWMAP SEEDBASE STAGES
  local TAG=$1 E2=$2 E3=$3 SB=$4 STG=$5 LD=$HOME/bench/dsv4pro-pc-$(date +%Y%m%d-%H%M)-$1
  log "sidecars a16 b12x-fix rowmap $E3"
  SC="E3DIR=$E3DIR B12X=\$HOME/dsv4pro-e3/b12x-fix E3_MODE=a16 E3ROWMAP=$E3 bash $E3DIR/launch/e3-sidecar.sh"
  ssh -o BatchMode=yes $R "$SC 1" > "$B/sidecar1-$TAG.txt" 2>&1 & P1=$!
  bash -c "$SC 0" > "$B/sidecar0-$TAG.txt" 2>&1; RC0=$?; wait $P1; RC1=$?
  log "sidecars rc $RC0/$RC1: $(tail -1 $B/sidecar0-$TAG.txt) | $(tail -1 $B/sidecar1-$TAG.txt)"
  if [ $RC0 = 0 ] && [ $RC1 = 0 ]; then
    log "boot $LD (E2 $E2, E3 $E3, E3_COUNT=1)"
    if ROWMAP=$E2 E3ROWMAP=$E3 E3DIR=$E3DIR bash $E3DIR/pc/pc-up.sh "$LD" > "$B/up-$TAG.txt" 2>&1; then
      log "$(tail -1 $B/up-$TAG.txt)"; grep -h "TIER WARNING" "$LD/server-rank0.log" | head -3 | while read -r l; do log "$l"; done
      bash $E3DIR/pc/pc-bench.sh "$LD" "$SB" "$STG" > "$B/bench-$TAG.txt" 2>&1; log "bench exit $?: $(grep -c rep "$LD/bench/summary.txt") lines"
    else log "BOOT FAILED: $(tail -3 $B/up-$TAG.txt | tr '\n' ' ')"; fi
  fi
  log "teardown rank 0 first"
  bash $E3DIR/launch/e3-down.sh sidecars >> "$B/master.log" 2>&1
  sudo -n rm -f /dev/shm/e3_dsv4pro_peer 2>/dev/null; ssh -o BatchMode=yes $R "sudo -n rm -f /dev/shm/e3_dsv4pro_peer 2>/dev/null"
  log "shmem L $(grep Shmem: /proc/meminfo) R $(ssh -o BatchMode=yes $R 'grep Shmem: /proc/meminfo')"
  log "$(gpus)"; echo "$LD" > "$B/logdir-$TAG.txt"
}
case $PH in
  base)
    log "stop dailies"
    docker stop glmf-daily qwen27b-rtx >/dev/null; ssh -o BatchMode=yes $R "docker stop dsv41-daily nemotron-super-rtx >/dev/null"
    sleep 5; log "$(gpus)"
    run_arm base rowmap-dsv4pro-acs-v1.json rowmap-e3-on-e2-acs-v1.json 8100 "real catid fit"
    touch "$B/BASEDONE" ;;
  refit)
    run_arm refit "${E2RM:?}" "${E3RM:?}" 8200 "real catid"
    touch "$B/REFITDONE" ;;
  restore)
    log "restore dailies (docker start, CDI)"
    docker start glmf-daily qwen27b-rtx >/dev/null 2>>"$B/master.log"; ssh -o BatchMode=yes $R "docker start dsv41-daily nemotron-super-rtx >/dev/null" 2>>"$B/master.log"
    log "started; $(docker ps --format '{{.Names}}:{{.Status}}' | tr '\n' ' ') | R $(ssh -o BatchMode=yes $R "docker ps --format '{{.Names}}:{{.Status}}'" | tr '\n' ' ')"
    touch "$B/ALLDONE" ;;
esac
log "phase done"
