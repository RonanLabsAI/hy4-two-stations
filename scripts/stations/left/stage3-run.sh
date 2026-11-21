#!/usr/bin/env bash
# L6-4h stage 3 (run on station-left): K- control, then A1 B1 B2 A2 (A = B + RTX PRO 6000 warm-expert sidecar).
# Usage: RM=<rowmap file in ~/hy4-e3/repo/rowmaps> WF=mxfp4|mxfp8 bash stage3-run.sh   Log ~/l6-4h/stage3.log, ends STAGE3_END.
W=$HOME/l6-4h; E=$HOME/hy4-e3/repo/launch/hy4; T=$HOME/hy4-e3/repo/tools; R=exx@192.168.101.2
H="python3 -I $W/tools/hy4/hy4bench.py"; BASE=http://127.0.0.1:8020
RM=${RM:?rowmap}; WF=${WF:?wfmt}; export MIN_COS=$([ "$WF" = mxfp4 ] && echo 0.97 || echo 0.995)
export CTX=163840 SEQS=16 OFFGB=166.5 EXTRA_ARGS="--max-logprobs 64 --enable-prompt-tokens-details"
B_C16=93.22   # stage-2 B C16 mean tok/s (runs/fp8b-20261121-010950-off166.5)
log() { echo "$(date '+%F %T') $*"; }
up() {  # $1 recipe $2 tag [VAR=val ...] -> OUT
  local rec=$1 tag=$2; shift 2
  OUT=$W/runs/s3-$tag-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; log "$tag: $rec load -> $OUT ($*)"
  env "$@" bash $E/hy4-up.sh $rec $OUT > $OUT/up.log 2>&1 || { tail -20 $OUT/up.log; log "$tag load FAILED"; return 1; }
  tail -8 $OUT/up.log
  ssh -o BatchMode=yes $R "cat $OUT/server-rank1.log" > $OUT/server-rank1.remote.log
  if grep -q "NET/Socket" $OUT/server-rank0.log $OUT/server-rank1.remote.log || ! grep -q "NET/IB" $OUT/server-rank0.log; then
    log "$tag: NCCL not on IB -> void"; return 1; fi
  grep -h "Total CPU offloaded parameters" $OUT/server-rank0.log $OUT/server-rank1.remote.log | tee $OUT/offload.txt
}
down() { bash $E/hy4-down.sh ${1:-} | tee $OUT/down.txt; sleep 15; }
sidecars_up() {
  E3ROWMAP=$RM WFMT=$WF bash $E/hy4-sidecar.sh 0 > $W/runs/sidecar0-$1.log 2>&1 &
  ssh -o BatchMode=yes $R "E3ROWMAP=$RM WFMT=$WF bash $E/hy4-sidecar.sh 1" > $W/runs/sidecar1-$1.log 2>&1 &
  wait; tail -3 $W/runs/sidecar0-$1.log $W/runs/sidecar1-$1.log
  grep -q "^serving" $HOME/hy4-e3/repo/logs/sidecar-rank0.log && ssh -o BatchMode=yes $R "grep -q '^serving' $HOME/hy4-e3/repo/logs/sidecar-rank1.log"
}
measure() {  # $1 tag, $2 = A|B (A adds quality + needle)
  $H speed   --base $BASE --engine vllm --model hy4 --rung $1 --out $OUT/bench --warm 1 --rounds 5 > $OUT/b-speed.log 2>&1; log "$1 speed rc=$?"
  $H prefill --base $BASE --engine vllm --model hy4 --rung $1 --out $OUT/bench > $OUT/b-prefill.log 2>&1; log "$1 prefill rc=$?"
  $H logprobs --base $BASE --engine vllm --model hy4 --rung $1 --out $OUT/bench > $OUT/b-lp.log 2>&1; log "$1 logprobs rc=$?"
  if [ "$2" = A ]; then
    $H quality --base $BASE --engine vllm --model hy4 --rung $1 --out $OUT/bench > $OUT/b-quality.log 2>&1; log "$1 quality rc=$?"
    $H needle  --base $BASE --engine vllm --model hy4 --rung $1 --out $OUT/bench > $OUT/b-needle.log 2>&1; log "$1 needle rc=$?"
  fi
  curl -s $BASE/metrics > $OUT/metrics-end.txt
}
stop3() { log "STOP: $*"; bash $E/hy4-down.sh sidecars > /dev/null 2>&1; echo STAGE3_END; exit 1; }
E3ENV="E3ROWMAP=$RM"

# --- K-: sensitivity control, replaced 2026-11-21 07:40 PT (prereg pre-freeze change 7): B with the drafter off (SPEC=0).
# Speed at both cells (3 rounds); gate = C1 ratio to stage-2 B C1 (27.61) < 0.95. C16 ratio reported, not gated.
B_C1=27.6083
up B kminus SPEC=0 || stop3 "K- load"
grep -q "spec=0" $OUT/e3-rank0.txt || stop3 "K- launched with the drafter on"
$H speed --base $BASE --engine vllm --model hy4 --rung kminus --out $OUT/bench --warm 1 --rounds 3 > $OUT/b-speed.log 2>&1
K1=$(python3 -c "import json; print(json.load(open(\"$OUT/bench/speed.summary.json\"))[\"c1\"][\"mean_tps\"])")
K16=$(python3 -c "import json; print(json.load(open(\"$OUT/bench/speed.summary.json\"))[\"c16\"][\"mean_tps\"])")
RATIO=$(python3 -c "print(round($K1/$B_C1,4))"); R16=$(python3 -c "print(round($K16/$B_C16,4))")
log "K- (SPEC=0) C1 $K1 tok/s ratio $RATIO (must be < 0.95); C16 $K16 ratio $R16 (reported)"; echo "c1 $RATIO c16 $R16" > $OUT/kminus-ratio.txt
down
python3 -c "import sys; sys.exit(0 if $RATIO < 0.95 else 1)" || stop3 "K- C1 ratio $RATIO >= 0.95: harness insensitive (prereg stop)"

# --- A1 (with cosine + identity gate before any timing) ---
sidecars_up A1 || stop3 "sidecars A1"
up A A1 $E3ENV E3_CHECK=4 E3_CHECK_MIN_T=1024 || stop3 "A1 load"
python3 $T/cos_gate.py send $BASE hy4 $HOME/r21/gsm8k_test.jsonl 6 > $OUT/gate-send.log 2>&1
sleep 12; ssh -o BatchMode=yes $R "cat $OUT/server-rank1.log" > $OUT/server-rank1.remote.log
python3 $T/cos_gate.py parse $OUT/server-rank0.log $OUT/server-rank1.remote.log $HOME/hy4-e3/repo/rowmaps/$RM > $OUT/gate.json 2>&1; G=$?
cat $OUT/gate.json | grep -E '"PASS"|cos_min|identity_fail|missing|timeouts' ; log "A1 gate rc=$G (MIN_COS=$MIN_COS)"
[ $G = 0 ] || { down sidecars; stop3 "A1 cosine/identity gate failed"; }
measure A1 A; down sidecars

# --- B1, B2 ---
up B B1 || stop3 "B1 load"; measure B1 B; down
up B B2 || stop3 "B2 load"; measure B2 B; down

# --- A2 ---
sidecars_up A2 || stop3 "sidecars A2"
up A A2 $E3ENV || stop3 "A2 load"; measure A2 A; down sidecars
for f in $W/runs/s3-A*-*/e3-stats-rank0.json; do echo "$f $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['peer'])" $f)"; done
echo STAGE3_END
