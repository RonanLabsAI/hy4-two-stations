#!/usr/bin/env bash
# L6-4h stage 1 runner. Usage: bash ~/l6-4h/stage1-run.sh left|right   Log ~/l6-4h/stage1-<st>.log, ends STAGE1_END.
ST=$1; W=$HOME/l6-4h; D=$W/tools/hy4/stage1
log() { echo "$(date '+%F %T') $*"; }
until grep -qE 'G0_PASS|G0_FAIL' $W/g0-$ST.log 2>/dev/null; do sleep 60; done
grep -q G0_PASS $W/g0-$ST.log || { log "G0 failed: no stage 1"; echo STAGE1_END; exit 1; }
until grep -q SHA_DONE $W/sha-check.log; do sleep 30; done
grep -q 'FAIL 0' $W/sha-check.log || { log "FP8 sha failures: stop"; echo STAGE1_END; exit 1; }
if [ $ST = left ]; then RUNGS="left-stq1_0 left-iq2"; else RUNGS="right-q4_k_m"; fi
for r in $RUNGS; do
  log "start $r"; L64H_APPROVED=1 bash $D/$r.sh > $W/runs/$r-$(date +%H%M).log 2>&1; rc=$?
  log "end $r rc=$rc"
done
echo STAGE1_END
