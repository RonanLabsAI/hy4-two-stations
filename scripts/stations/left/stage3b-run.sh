#!/usr/bin/env bash
# L6-4h stage 3b follow-ups (prereg pre-freeze change 8, 2026-11-21 12:45 PT). Run on station-left:
#   bash stage3b-run.sh   Log ~/l6-4h/stage3b.log, ends STAGE3B_END.
# Order: A4 (MXFP4, held-out speed) | A8-1 (MXFP8, gate + full) | B3 | B4 | A8-2 (MXFP8 full) | A4b (MXFP4, held-out speed).
# An MXFP8 failure (sidecar, load or gate) skips the remaining MXFP8 loads but never the held-out bracket.
W=$HOME/l6-4h; E=$HOME/hy4-e3/repo/launch/hy4; T=$HOME/hy4-e3/repo/tools; R=exx@192.168.101.2
H="python3 -I $W/tools/hy4/hy4bench.py"; BASE=http://127.0.0.1:8020
export CTX=163840 SEQS=16 OFFGB=166.5 EXTRA_ARGS="--max-logprobs 64 --enable-prompt-tokens-details"
RM4=rowmap-hy4-v1-mxfp4.json; RM8=rowmap-hy4-v1-mxfp8.json; MX8=1
log() { echo "$(date '+%F %T') $*"; }
up() {  # $1 recipe $2 tag [VAR=val ...] -> OUT
  local rec=$1 tag=$2; shift 2
  OUT=$W/runs/s3b-$tag-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; log "$tag: $rec load -> $OUT ($*)"
  env "$@" bash $E/hy4-up.sh $rec $OUT > $OUT/up.log 2>&1 || { tail -20 $OUT/up.log; log "$tag load FAILED"; return 1; }
  tail -8 $OUT/up.log
  ssh -o BatchMode=yes $R "cat $OUT/server-rank1.log" > $OUT/server-rank1.remote.log
  if grep -q "NET/Socket" $OUT/server-rank0.log $OUT/server-rank1.remote.log || ! grep -q "NET/IB" $OUT/server-rank0.log; then
    log "$tag: NCCL not on IB -> void"; return 1; fi
  grep -h "Total CPU offloaded parameters" $OUT/server-rank0.log $OUT/server-rank1.remote.log | tee $OUT/offload.txt
}
down() { bash $E/hy4-down.sh ${1:-} | tee $OUT/down.txt; sleep 15; }
sidecars_up() {  # $1 tag $2 rowmap $3 wfmt
  E3ROWMAP=$2 WFMT=$3 bash $E/hy4-sidecar.sh 0 > $W/runs/sidecar0-3b-$1.log 2>&1 &
  ssh -o BatchMode=yes $R "E3ROWMAP=$2 WFMT=$3 bash $E/hy4-sidecar.sh 1" > $W/runs/sidecar1-3b-$1.log 2>&1 &
  wait; tail -3 $W/runs/sidecar0-3b-$1.log $W/runs/sidecar1-3b-$1.log
  grep -q "^serving" $HOME/hy4-e3/repo/logs/sidecar-rank0.log && ssh -o BatchMode=yes $R "grep -q '^serving' $HOME/hy4-e3/repo/logs/sidecar-rank1.log"
}
sp()  { $H speed --base $BASE --engine vllm --model hy4 --rung $1 --out $OUT/bench --warm 1 --rounds 5 > $OUT/b-speed.log 2>&1; log "$1 speed rc=$?"; }
spho(){ $H speed --base $BASE --engine vllm --model hy4 --rung $1 --out $OUT/bench --warm 1 --rounds 5 --speed-prompts speed_heldout.json > $OUT/b-speed-heldout.log 2>&1; log "$1 speed-heldout rc=$?"; }
pf()  { $H prefill --base $BASE --engine vllm --model hy4 --rung $1 --out $OUT/bench > $OUT/b-prefill.log 2>&1; log "$1 prefill rc=$?"; }
afull(){ sp $1; spho $1; pf $1
  $H logprobs --base $BASE --engine vllm --model hy4 --rung $1 --out $OUT/bench > $OUT/b-lp.log 2>&1; log "$1 logprobs rc=$?"
  $H quality  --base $BASE --engine vllm --model hy4 --rung $1 --out $OUT/bench > $OUT/b-quality.log 2>&1; log "$1 quality rc=$?"
  $H needle   --base $BASE --engine vllm --model hy4 --rung $1 --out $OUT/bench > $OUT/b-needle.log 2>&1; log "$1 needle rc=$?"
  curl -s $BASE/metrics > $OUT/metrics-end.txt; }
fin() { curl -s $BASE/metrics > $OUT/metrics-end.txt; }
stop3() { log "STOP: $*"; bash $E/hy4-down.sh sidecars > /dev/null 2>&1; echo STAGE3B_END; exit 1; }

# --- A4: MXFP4, held-out speed only ---
sidecars_up A4 $RM4 mxfp4 || stop3 "sidecars A4"
up A A4 E3ROWMAP=$RM4 || stop3 "A4 load"; spho A4; fin; down sidecars

# --- A8-1: MXFP8 with gate ---
export MIN_COS=0.995
if sidecars_up A8-1 $RM8 mxfp8 && up A A8-1 E3ROWMAP=$RM8 E3_CHECK=4 E3_CHECK_MIN_T=1024; then
  python3 $T/cos_gate.py send $BASE hy4 $HOME/r21/gsm8k_test.jsonl 6 > $OUT/gate-send.log 2>&1
  sleep 12; ssh -o BatchMode=yes $R "cat $OUT/server-rank1.log" > $OUT/server-rank1.remote.log
  python3 $T/cos_gate.py parse $OUT/server-rank0.log $OUT/server-rank1.remote.log $HOME/hy4-e3/repo/rowmaps/$RM8 > $OUT/gate.json 2>&1; G=$?
  grep -E '"PASS"|cos_min|identity_fail|missing|timeouts' $OUT/gate.json; log "A8-1 gate rc=$G (MIN_COS=$MIN_COS)"
  if [ $G = 0 ]; then afull A8-1; else MX8=0; log "A8-1 gate failed: MXFP8 block skipped"; fi
  down sidecars
else MX8=0; log "A8-1 sidecar/load failed: MXFP8 block skipped"; [ -n "${OUT:-}" ] && down sidecars; fi

# --- B3, B4 ---
up B B3 || stop3 "B3 load"; sp B3; spho B3; pf B3; fin; down
up B B4 || stop3 "B4 load"; sp B4; spho B4; pf B4; fin; down

# --- A8-2 ---
if [ $MX8 = 1 ]; then
  if sidecars_up A8-2 $RM8 mxfp8 && up A A8-2 E3ROWMAP=$RM8; then afull A8-2; else log "A8-2 failed"; fi; down sidecars
fi

# --- A4b ---
sidecars_up A4b $RM4 mxfp4 || stop3 "sidecars A4b"
up A A4b E3ROWMAP=$RM4 || stop3 "A4b load"; spho A4b; fin; down sidecars
for f in $W/runs/s3b-A*-*/e3-stats-rank0.json; do echo "$f $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['peer'])" $f)"; done
echo STAGE3B_END
