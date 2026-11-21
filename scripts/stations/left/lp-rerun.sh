#!/usr/bin/env bash
# Re-run only the logprob capture for one stage-1 rung (prereg pre-freeze change 4). Usage: lp-rerun.sh <rung-script> <done-glob>
# Waits for this Station's stage-1 runner to end, skips if <done-glob> already has a logprobs.summary.json.
W=$HOME/l6-4h; D=$W/tools/hy4/stage1; R=$1; DONE=$2; ST=${R%%-*}
until grep -q STAGE1_END $W/stage1-$ST.log; do sleep 60; done
if ls $W/runs/$DONE/bench/logprobs.summary.json >/dev/null 2>&1; then echo "$(date '+%F %T') $R logprobs already complete; skip"; echo LP_RERUN_END; exit 0; fi
sed 's/^measure.*$/python3 -I "$HARNESS" logprobs --base "$BASE" --engine llamacpp --rung "$RUNG" --out "$OUT\/bench" 2>\&1 | tee "$OUT\/bench.log"/' $D/$R.sh > $D/lp-$R.sh
grep -n 'logprobs --base' $D/lp-$R.sh
echo "$(date '+%F %T') start lp rerun $R"; L64H_APPROVED=1 bash $D/lp-$R.sh > $W/runs/lp-$R-$(date +%H%M).log 2>&1; echo "$(date '+%F %T') end lp rerun $R rc=$?"
echo LP_RERUN_END
