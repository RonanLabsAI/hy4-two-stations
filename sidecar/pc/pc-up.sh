#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# pc-up.sh LOGDIR -- PC (2026-10-09): r31/r31-up.sh with E3DIR passthrough and E3_COUNT=1, no SPEC, no nsys (E3-A16 recipe).
# env: ROWMAP (E2, in ~/dsv4pro/prof) E3ROWMAP (in $E3DIR/rowmaps) E3DIR (default ~/pc-1009/repo)
set -euo pipefail
LOGDIR=${1:?logdir}; R=exx@192.168.101.2; L=$HOME/r31/launch-r31.sh
V=(LOGDIR="$LOGDIR" PIN_MODE=split ROWMAP="${ROWMAP:-rowmap-dsv4pro-acs-v1.json}" E3HOOK=1 E3_ENABLE=1
   E3ROWMAP="${E3ROWMAP:-rowmap-e3-on-e2-acs-v1.json}" E3DIR="${E3DIR:-$HOME/pc-1009/repo}"
   E3_CHECK=0 E3_CHECK_MIN_T=1024 E3_COUNT=1 SPEC= NSYS_TAG= UTIL="${UTIL:-0.95}" CTX="${CTX:-131072}" SEQS=16 EXTRA_ARGS=)
Q=$(printf ' %q' "${V[@]}")
mkdir -p "$LOGDIR"; ssh -o BatchMode=yes $R "mkdir -p $LOGDIR"
ssh -o BatchMode=yes $R "env$Q bash $L 1" | tee "$LOGDIR/up-rank1.txt"
sleep 5
env "${V[@]}" bash "$L" 0 | tee "$LOGDIR/up-rank0.txt"
T0=$(date +%s)
until curl -sf -m 5 http://127.0.0.1:8010/health >/dev/null 2>&1; do
  docker ps -q -f name='^dsv4pro-e3$' | grep -q . || { echo "rank0 container died"; tail -60 "$LOGDIR/server-rank0.log"; exit 1; }
  ssh -o BatchMode=yes $R "docker ps -q -f name='^dsv4pro-e3\$'" | grep -q . || { echo "rank1 container died"; ssh $R "tail -60 $LOGDIR/server-rank1.log"; exit 1; }
  [ $(( $(date +%s) - T0 )) -gt 2700 ] && { echo "not ready after 45 min"; exit 1; }
  sleep 15
done
echo "ready $(date '+%F %T') after $(( $(date +%s) - T0 )) s"
