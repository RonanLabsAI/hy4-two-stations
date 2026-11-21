#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# hy4-up.sh RECIPE LOGDIR -- run on station-left: start rank 1 (station-right, over rail 1) then rank 0, wait for the API.
#   RECIPE  B      E3HOOK=0 (GB300 + Grace cold experts)
#           COUNT  B + routing-count capture (E3HOOK=1 E3_MODE=count): the input for the first Hy4 rowmap
#           A      B + 6000 warm experts (E3HOOK=1 E3_MODE=serve); the sidecars must already be serving on both Stations
#                  (launch/hy4/hy4-sidecar.sh 0 here and 1 on station-right, same E3ROWMAP/WFMT)
# Pass-through env (forwarded to BOTH ranks): OFFGB CTX SEQS MBT UTIL LMO SPEC DD PIN MOE MODEL IMAGE PORT E3DIR E3_SHM
#   E3ROWMAP E3_CHECK E3_CHECK_MIN_T E3_COUNT E3_ENABLE E3_PEER_SPINS EXTRA_ARGS
# After the API answers: A / COUNT fail unless both ranks logged "E3 patched ... mode=<serve|count>" (sitecustomize
# swallows hook-install errors, so a broken hook would otherwise serve stock B numbers). COUNT also snapshots the counts
# at ready (e3-counts-ready-rank*.json): the profile run and warm-ups are in it, so build rowmaps from the DELTA
# (tools/counts_delta.py e3-counts-ready-rankN.json e3-counts-rankN.json out.json).
set -euo pipefail
RECIPE=${1:?B|COUNT|A}; LOGDIR=${2:?logdir}; R=exx@192.168.101.2; L=$HOME/hy4-e3/repo/launch/hy4/launch-hy4.sh
case "$RECIPE" in
  B) HOOK="E3HOOK=0" ;;
  COUNT) HOOK="E3HOOK=1 E3_MODE=count" ;;
  A) HOOK="E3HOOK=1 E3_MODE=serve" ;;
  *) echo "RECIPE must be B|COUNT|A"; exit 2 ;;
esac
PASS="LOGDIR=$LOGDIR $HOOK"
for v in OFFGB CTX SEQS MBT UTIL LMO SPEC DD PIN MOE MODEL IMAGE PORT E3DIR E3_SHM E3ROWMAP E3_CHECK E3_CHECK_MIN_T \
         E3_COUNT E3_ENABLE E3_PEER_SPINS NCCL_PROTO; do
  [ -n "${!v:-}" ] && PASS="$PASS $v=${!v}"
done
[ -n "${EXTRA_ARGS:-}" ] && PASS="$PASS EXTRA_ARGS='$EXTRA_ARGS'"
mkdir -p "$LOGDIR"; ssh -o BatchMode=yes $R "mkdir -p $LOGDIR"
echo "$RECIPE: $PASS" | tee "$LOGDIR/recipe.txt"
ssh -o BatchMode=yes $R "env $PASS bash $L 1" | tee "$LOGDIR/up-rank1.txt"
sleep 5
eval "env $PASS bash $L 0" | tee "$LOGDIR/up-rank0.txt"
T0=$(date +%s)
until curl -sf http://127.0.0.1:${PORT:-8020}/v1/models >/dev/null 2>&1; do
  docker ps -q -f name='^hy4-ep2$' | grep -q . || { echo "rank0 container died"; tail -40 "$LOGDIR/server-rank0.log"; exit 1; }
  ssh -o BatchMode=yes $R "docker ps -q -f name='^hy4-ep2\$'" | grep -q . || { echo "rank1 container died"; ssh $R "tail -40 $LOGDIR/server-rank1.log"; exit 1; }
  [ $(( $(date +%s) - T0 )) -gt 3600 ] && { echo "not ready after 60 min"; exit 1; }
  sleep 15
done
echo "ready $(date '+%F %T') after $(( $(date +%s) - T0 )) s"
if [ "$RECIPE" != B ]; then
  MODE=$([ "$RECIPE" = A ] && echo serve || echo count)
  ssh -o BatchMode=yes $R "cat $LOGDIR/server-rank1.log" > "$LOGDIR/server-rank1.remote.log"
  for f in "$LOGDIR/server-rank0.log" "$LOGDIR/server-rank1.remote.log"; do
    grep -q "E3 hook FAILED" "$f" && { echo "E3 hook FAILED to install ($f): this is a stock-B server"; grep "E3 hook FAILED" "$f" | head -3; exit 1; }
    grep -q "E3 patched .*mode=$MODE" "$f" || { echo "no 'E3 patched ... mode=$MODE' in $f: hook not active"; exit 1; }
  done
  echo "E3 hook active (mode=$MODE) on both ranks"
  if [ "$RECIPE" = A ]; then
    # Review 2026-11-21: the "patched" line only proves the import. If layer names do not match, every layer falls
    # back to stock forward and A serves B numbers. Require each rank to have masked >= the rowmap's warm layers.
    RM=${E3DIR:-$HOME/hy4-e3/repo}/rowmaps/${E3ROWMAP:-rowmap-hy4-prior-mxfp4.json}
    WANT=$(python3 -c "import json,sys; print(sum(1 for v in json.load(open(sys.argv[1]))['layers'].values() if v.get('warm')))" "$RM")
    sleep 12   # stats dump every 10 s; warm-up requests have set every layer up by now
    for r in 0 1; do
      c="cat $LOGDIR/e3-stats-rank$r.json"
      got=$( { if [ $r = 0 ]; then $c; else ssh -o BatchMode=yes $R "$c"; fi; } 2>/dev/null \
             | python3 -c "import json,sys; print(json.load(sys.stdin).get('setup_warm_layers', -1))" 2>/dev/null || echo -1)
      [ "${got:--1}" -ge "$WANT" ] || { echo "rank$r masked $got warm layers < rowmap $WANT: hook not active per layer"; exit 1; }
      echo "rank$r: $got warm layers masked (rowmap $WANT)"
    done
  fi
  if [ "$RECIPE" = COUNT ]; then
    sleep 10   # counts reach the host copy within E3_COUNT_COPY_SEC + E3_COUNT_DUMP_SEC (1 + 3 s) of an eager call
    for r in 0 1; do
      c="cp $LOGDIR/e3-counts-rank$r.json $LOGDIR/e3-counts-ready-rank$r.json"
      if [ $r = 0 ]; then $c; else ssh -o BatchMode=yes $R "$c"; fi
    done
    echo "count baseline at ready: $LOGDIR/e3-counts-ready-rank{0,1}.json"
  fi
fi
grep -h "Total CPU offloaded parameters\|Model loading took\|KV cache\|moe backend\|Using .* MoE backend\|E3 " "$LOGDIR"/server-rank0.log | head -20
