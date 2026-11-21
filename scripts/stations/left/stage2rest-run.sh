#!/usr/bin/env bash
# L6-4h stage 2 remainder (run on station-left): D512 load (drafter at 2K-512K on one placement), then COUNT load
# (routing counts for the stage-3 rowmap under frozen harness traffic). Log ~/l6-4h/stage2rest.log, ends STAGE2REST_END.
W=$HOME/l6-4h; E=$HOME/hy4-e3/repo/launch/hy4; T=$HOME/hy4-e3/repo/tools; R=exx@192.168.101.2
H="python3 -I $W/tools/hy4/hy4bench.py"; BASE=http://127.0.0.1:8020
log() { echo "$(date '+%F %T') $*"; }
up_ladder() {  # $1 recipe, $2 tag, rest: offload steps. Sets OUT.
  local rec=$1 tag=$2; shift 2; OUT=""
  for OFF in "$@"; do
    OUT=$W/runs/$tag-$(date +%Y%m%d-%H%M%S)-off$OFF; mkdir -p $OUT; log "$tag load OFFGB=$OFF -> $OUT"
    OFFGB=$OFF bash $E/hy4-up.sh $rec $OUT > $OUT/up.log 2>&1 && { tail -6 $OUT/up.log; return 0; }
    tail -15 $OUT/up.log
    if grep -qiE "out of memory|No available memory|KV cache|max_model_len|larger than the maximum" $OUT/server-rank0.log $OUT/up.log 2>/dev/null; then
      log "$tag OFFGB=$OFF failed on memory; next step"; bash $E/hy4-down.sh; sleep 20; continue; fi
    log "$tag failed (not memory)"; bash $E/hy4-down.sh; OUT=""; return 1
  done
  log "$tag did not fit"; OUT=""; return 1
}
ib_ok() { ssh -o BatchMode=yes $R "cat $OUT/server-rank1.log" > $OUT/server-rank1.remote.log
  grep -h "Total CPU offloaded parameters" $OUT/server-rank0.log $OUT/server-rank1.remote.log | tee $OUT/offload.txt
  grep -hiE "GPU KV cache size" $OUT/server-rank0.log | tail -1 | tee $OUT/kv.txt
  ! grep -q "NET/Socket" $OUT/server-rank0.log $OUT/server-rank1.remote.log && grep -q "NET/IB" $OUT/server-rank0.log; }

# --- D512 ---
export CTX=532480 SEQS=4 EXTRA_ARGS="--max-logprobs 64 --enable-prompt-tokens-details"
if up_ladder B fp8d512 193.5 198.0 202.5 && ib_ok; then
  $H check --base $BASE --engine vllm --model hy4 --rung fp8d512 --out $OUT/bench > $OUT/bench-check.log 2>&1; rc=$?
  log "D512 check rc=$rc"
  [ $rc = 0 ] && { $H drafter --base $BASE --engine vllm --model hy4 --rung fp8d512 --out $OUT/bench > $OUT/bench.log 2>&1; log "D512 drafter rc=$?"; }
  curl -s $BASE/metrics > $OUT/metrics-end.txt
else log "D512 unrun or void"; fi
bash $E/hy4-down.sh | tee -a $W/stage2rest.down.txt; sleep 15

# --- COUNT ---
export CTX=163840 SEQS=16 EXTRA_ARGS="--max-logprobs 64 --enable-prompt-tokens-details"
if up_ladder COUNT fp8count 166.5 && ib_ok; then
  $H speed --base $BASE --engine vllm --model hy4 --rung fp8count --out $OUT/bench --rounds 1 --warm 0 > $OUT/bench-speed.log 2>&1
  log "COUNT speed rc=$?"
  $H quality --base $BASE --engine vllm --model hy4 --rung fp8count --out $OUT/bench --limit 100 > $OUT/bench-quality.log 2>&1
  log "COUNT quality rc=$?"
  sleep 15
  python3 $T/counts_delta.py $OUT/e3-counts-ready-rank0.json $OUT/e3-counts-rank0.json $OUT/delta-rank0.json; log "delta rank0 rc=$?"
  ssh -o BatchMode=yes $R "python3 $T/counts_delta.py $OUT/e3-counts-ready-rank1.json $OUT/e3-counts-rank1.json $OUT/delta-rank1.json"; log "delta rank1 rc=$?"
  scp -q $R:$OUT/delta-rank1.json $OUT/ ; ls -la $OUT/delta-rank*.json
else log "COUNT unrun or void"; fi
bash $E/hy4-down.sh | tee -a $W/stage2rest.down.txt
echo STAGE2REST_END
