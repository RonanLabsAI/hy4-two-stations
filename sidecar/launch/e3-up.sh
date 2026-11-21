#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# e3-up.sh LOGDIR -- run on .94: start rank 1 (.95, via rail-1 ssh) then rank 0 (.94) with launch-dsv4pro-e3.sh, wait
# until the API answers. Pass-through env: E3HOOK E3_ENABLE E3_CHECK E3_CHECK_MIN_T E3_COUNT E3ROWMAP.
# Sidecars (E3HOOK=1) must already be serving on both Stations (launch/e3-sidecar.sh 0|1).
set -euo pipefail
LOGDIR=${1:?logdir}; R=exx@192.168.101.2; L=$HOME/dsv4pro-e3/repo/launch/launch-dsv4pro-e3.sh
PASS="LOGDIR=$LOGDIR E3HOOK=${E3HOOK:-1} E3_ENABLE=${E3_ENABLE:-1} E3_CHECK=${E3_CHECK:-0} E3_CHECK_MIN_T=${E3_CHECK_MIN_T:-1024} E3_COUNT=${E3_COUNT:-0} E3ROWMAP=${E3ROWMAP:-rowmap-e3-gsm-v1.json}${PIN_MODE:+ PIN_MODE=$PIN_MODE ROWMAP=$ROWMAP}"
mkdir -p "$LOGDIR"; ssh -o BatchMode=yes $R "mkdir -p $LOGDIR"
ssh -o BatchMode=yes $R "env $PASS bash $L 1" | tee "$LOGDIR/up-rank1.txt"
sleep 5
env $PASS bash "$L" 0 | tee "$LOGDIR/up-rank0.txt"
T0=$(date +%s)
until curl -sf http://127.0.0.1:8010/v1/models >/dev/null 2>&1; do
  docker ps -q -f name='^dsv4pro-e3$' | grep -q . || { echo "rank0 container died"; tail -40 "$LOGDIR/server-rank0.log"; exit 1; }
  ssh -o BatchMode=yes $R "docker ps -q -f name='^dsv4pro-e3\$'" | grep -q . || { echo "rank1 container died"; ssh $R "tail -40 $LOGDIR/server-rank1.log"; exit 1; }
  [ $(( $(date +%s) - T0 )) -gt 2700 ] && { echo "not ready after 45 min"; exit 1; }
  sleep 15
done
echo "ready $(date '+%F %T') after $(( $(date +%s) - T0 )) s"
