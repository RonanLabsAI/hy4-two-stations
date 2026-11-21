#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# r31b-up.sh LOGDIR -- R3.1b copy of r31/r31-up.sh (md5 c0c05c3b) launching r31b/launch-r31b.sh. Run on .94: rank 1 (.95 over rail 1) then rank 0 with r31/launch-r31.sh (E3-A16 recipe), wait for /health.
# Fixed E3-A16 recipe: PIN_MODE=split ROWMAP=rowmap-dsv4pro-acs-v1.json E3ROWMAP=rowmap-e3-on-e2-acs-v1.json (sidecars must be
# serving with E3_MODE=a16, b12x fork dce2933e). Pass-through: SPEC NSYS_TAG UTIL CTX SEQS E3_CHECK E3_CHECK_MIN_T E3_COUNT EXTRA_ARGS.
set -euo pipefail
LOGDIR=${1:?logdir}; R=exx@192.168.101.2; L=$HOME/r31b/repo/r31b/launch-r31b.sh
V=(LOGDIR="$LOGDIR" PIN_MODE=split ROWMAP=rowmap-dsv4pro-acs-v1.json E3HOOK=1 E3_ENABLE=1 E3ROWMAP=rowmap-e3-on-e2-acs-v1.json
   E3_CHECK="${E3_CHECK:-0}" E3_CHECK_MIN_T="${E3_CHECK_MIN_T:-1024}" E3_COUNT="${E3_COUNT:-0}"
   SPEC="${SPEC:-}" NSYS_TAG="${NSYS_TAG:-}" UTIL="${UTIL:-0.95}" CTX="${CTX:-131072}" SEQS="${SEQS:-16}" EXTRA_ARGS="${EXTRA_ARGS:-}"
   GUARD="${GUARD:-1}" R31B_FAILFAST="${R31B_FAILFAST:-1}" R31B_DRAFT_DG_WARMUP="${R31B_DRAFT_DG_WARMUP:-relax}" R31B_INJECT="${R31B_INJECT:-}"
   R31B_KCTL="${R31B_KCTL:-0}" R31B_KCTL_KOFF="${R31B_KCTL_KOFF:-1}" R31B_KCTL_OFF="${R31B_KCTL_OFF:-1.5}" R31B_KCTL_ON="${R31B_KCTL_ON:-1.9}"
   R31B_KCTL_PROBE="${R31B_KCTL_PROBE:-16}" R31C_SKIPDRAFT="${R31C_SKIPDRAFT:-0}" R31C_PROBE_MAX="${R31C_PROBE_MAX:-256}" R31C_CHECK_EVERY="${R31C_CHECK_EVERY:-0}"
   EXEC_TIMEOUT="${EXEC_TIMEOUT:-120}" KVBYTES="${KVBYTES:-}" MBT="${MBT:-8192}")
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
