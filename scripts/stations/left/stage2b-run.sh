#!/usr/bin/env bash
# L6-4h stage 2, load B (FP8, EP2 over both Stations, Grace cold experts, DFly2). Run on station-left.
# Log ~/l6-4h/stage2b.log, ends STAGE2B_END. Waits for stage-1 work (runners, logprob reruns) to finish first.
W=$HOME/l6-4h; E=$HOME/hy4-e3/repo/launch/hy4; R=exx@192.168.101.2
log() { echo "$(date '+%F %T') $*"; }
while pgrep -f '^bash (stage1-run|lp-rerun).sh' >/dev/null || ssh -o BatchMode=yes $R "pgrep -f '^bash (stage1-run|lp-rerun).sh'" >/dev/null; do sleep 60; done
log "stage-1 work finished on both Stations"
export CTX=163840 SEQS=16 EXTRA_ARGS="--max-logprobs 64 --enable-prompt-tokens-details"
OUT=""; for OFF in 166.5 171.0 175.5; do
  OUT=$W/runs/fp8b-$(date +%Y%m%d-%H%M%S)-off$OFF; mkdir -p $OUT
  log "B load OFFGB=$OFF -> $OUT"
  OFFGB=$OFF bash $E/hy4-up.sh B $OUT > $OUT/up.log 2>&1; rc=$?
  tail -25 $OUT/up.log
  [ $rc = 0 ] && break
  if grep -qiE "out of memory|No available memory|KV cache|max_model_len|larger than the maximum" $OUT/server-rank0.log $OUT/up.log 2>/dev/null; then
    log "B load OFFGB=$OFF failed on memory; fallback step"; bash $E/hy4-down.sh; sleep 20; OUT=""; continue
  fi
  log "B load failed (not memory): stop"; bash $E/hy4-down.sh; echo STAGE2B_END; exit 1
done
[ -n "$OUT" ] || { log "B did not fit after 2 fallback steps: unrun"; echo STAGE2B_END; exit 1; }
ssh -o BatchMode=yes $R "cat $OUT/server-rank1.log" > $OUT/server-rank1.remote.log
grep -h "Total CPU offloaded parameters" $OUT/server-rank0.log $OUT/server-rank1.remote.log | tee $OUT/offload.txt
if grep -q "NET/Socket" $OUT/server-rank0.log $OUT/server-rank1.remote.log || ! grep -q "NET/IB" $OUT/server-rank0.log; then
  log "NCCL not on IB: load void (prereg stop)"; bash $E/hy4-down.sh; echo STAGE2B_END; exit 1; fi
grep -hiE "GPU KV cache size|Maximum concurrency|KV cache" $OUT/server-rank0.log | tail -4 | tee $OUT/kv.txt
nvidia-smi --query-gpu=timestamp,uuid,clocks.sm,power.draw,temperature.gpu,memory.used,utilization.gpu --format=csv -l 2 > $OUT/gpu-left-2s.csv & SL=$!
ssh -o BatchMode=yes $R "nohup nvidia-smi --query-gpu=timestamp,uuid,clocks.sm,power.draw,temperature.gpu,memory.used,utilization.gpu --format=csv -l 2 > $OUT/gpu-right-2s.csv 2>&1 & echo \$! > $OUT/sampler-right.pid"
log "harness start"
python3 -I $W/tools/hy4/hy4bench.py all --base http://127.0.0.1:8020 --engine vllm --model hy4 --rung fp8b \
  --out $OUT/bench --think --max-draft 131072 > $OUT/bench.log 2>&1; rc=$?
log "harness end rc=$rc"; tail -5 $OUT/bench.log
curl -s http://127.0.0.1:8020/metrics > $OUT/metrics-end.txt
kill $SL; ssh -o BatchMode=yes $R "kill \$(cat $OUT/sampler-right.pid)"
bash $E/hy4-down.sh | tee $OUT/down.txt
echo STAGE2B_END
