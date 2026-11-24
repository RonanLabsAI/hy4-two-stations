#!/usr/bin/env bash
# L6-4g window run (station-left). Usage: bash ~/l6-4g/run.sh gn|gb|kday|gnt|hat|gnc|hac|gnc64|hac64   Log ~/l6-4g/run.log, each phase ends
# with "<PHASE>_END". Arms (prereg docs/l6/prereg-l6-4g-hy4-vs-glm53.md): G-N GLM-5.3 NVFP4 SGLang; G-B GLM-5.3 FP8 vLLM
# (the L6-4h B recipe); K-day Hy4 B speed only. Dailies must already be stopped by window.sh (this script checks).
set -uo pipefail
W=$HOME/l6-4g; L=$W/launch; R=exx@192.168.101.2; BASE=http://127.0.0.1:8020
H="python3 -I $W/tools/hy4/hy4bench.py"; exec >> $W/run.log 2>&1
log() { echo "$(TZ=America/Los_Angeles date '+%F %T %Z') $*"; }
idle() { for c in "nvidia-smi --query-gpu=name,memory.used --format=csv,noheader" ; do
  l=$($c | awk -F', ' '/GB300/{print $2+0}'); r=$(ssh -o BatchMode=yes $R "$c" | awk -F', ' '/GB300/{print $2+0}')
  [ "$l" -lt 8192 ] && [ "$r" -lt 8192 ] || { log "GB300 not idle (left $l MiB, right $r MiB): dailies running? STOP"; return 1; }; done; }
nccl_ib() {  # $1 OUT
  ssh -o BatchMode=yes $R "cat $1/server-rank1.log" > $1/server-rank1.remote.log
  if grep -q "NET/Socket" $1/server-rank0.log $1/server-rank1.remote.log || ! grep -q "NET/IB" $1/server-rank0.log; then
    log "NCCL not on IB -> load void"; return 1; fi; }
bench() {  # $1 rung $2 engine $3 served-model $4 frozen dir $5... cmds
  local rung=$1 eng=$2 mdl=$3 fz=$4; shift 4
  for cmd in "$@"; do
    case $cmd in
      speed-heldout) $H speed --base $BASE --engine $eng --model $mdl --rung $rung --frozen $fz --out $OUT/bench \
                       --speed-prompts speed_heldout.json --warm 1 --rounds 5 > $OUT/b-$cmd.log 2>&1 ;;
      think) $H quality --base $BASE --engine $eng --model $mdl --rung $rung --frozen $fz --out $OUT/bench \
               --think-file mmlu_pro_think.json --think-cap 16384 --think-inflight 8 > $OUT/b-$cmd.log 2>&1 ;;
      code) $H code --base $BASE --engine $eng --model $mdl --rung $rung --frozen $fz --out $OUT/bench \
              --code-cap ${CODECAP:-32768} --think-inflight 8 > $OUT/b-$cmd.log 2>&1 ;;
      speed) $H speed --base $BASE --engine $eng --model $mdl --rung $rung --frozen $fz --out $OUT/bench --warm 1 --rounds 5 > $OUT/b-$cmd.log 2>&1 ;;
      *) $H $cmd --base $BASE --engine $eng --model $mdl --rung $rung --frozen $fz --out $OUT/bench > $OUT/b-$cmd.log 2>&1 ;;
    esac
    rc=$?; log "$rung $cmd rc=$rc"
    if [ $cmd = check ] && [ $rc != 0 ]; then log "$rung check gate failed: nothing else runs"; tail -5 $OUT/b-check.log; return 1; fi
  done
  curl -s $BASE/metrics > $OUT/metrics-end.txt
}
PHASE=${1:?gn|gb|kday|gnt|hat|gnc|hac|gnc64|hac64}; case $PHASE in *64) export CODECAP=65536;; esac; idle || { echo ${PHASE^^}_END; exit 1; }
OUT=$W/runs/$PHASE-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; log "$PHASE start -> $OUT"
case $PHASE in
  gn)
    grep -q "glm53-nvfp4 PASS" $W/hashcheck.log && ssh -o BatchMode=yes $R "grep -q 'glm53-nvfp4 PASS' $W/hashcheck.log" \
      || { log "NVFP4 hash check not PASS on both Stations: no load"; echo GN_END; exit 1; }
    bash $L/glm53n-up.sh $OUT > $OUT/up.log 2>&1 || { tail -20 $OUT/up.log; log "G-N load FAILED"; bash $L/glm53n-down.sh; echo GN_END; exit 1; }
    tail -2 $OUT/up.log; nccl_ib $OUT || { bash $L/glm53n-down.sh; echo GN_END; exit 1; }
    bench gn sglang glm-5.3 frozen-glm53 check speed speed-heldout prefill needle quality
    bash $L/glm53n-down.sh | tee $OUT/down.txt; echo GN_END ;;
  gb)
    grep -q "glm53-fp8 PASS" $W/hashcheck.log && ssh -o BatchMode=yes $R "grep -q 'glm53-fp8 PASS' $W/hashcheck.log" \
      || { log "FP8 hash check not PASS on both Stations: no load"; echo GB_END; exit 1; }
    OFF=${OFFGB:-148.5}
    # load attempt 1 (12:30 PT) failed: FLASHMLA_SPARSE rejects kv_cache_dtype fp8_e4m3, which vLLM picked for GLM's
    # "auto"; Hy4 B's "auto" resolved to bf16, so G-B pins bfloat16 to match it (deviation, recorded).
    CTX=163840 SEQS=16 OFFGB=$OFF EXTRA_ARGS="--max-logprobs 64 --enable-prompt-tokens-details --kv-cache-dtype bfloat16" \
      bash $L/glm53-up.sh B $OUT > $OUT/up.log 2>&1 || { tail -30 $OUT/up.log; log "G-B load FAILED (OFFGB $OFF)"; bash $HOME/hy4-e3/repo/launch/hy4/hy4-down.sh; echo GB_END; exit 1; }
    tail -8 $OUT/up.log; nccl_ib $OUT || { bash $HOME/hy4-e3/repo/launch/hy4/hy4-down.sh; echo GB_END; exit 1; }
    got=$(grep -ho "Total CPU offloaded parameters: [0-9.]*" $OUT/server-rank0.log | awk '{print $NF}' | head -1)
    log "offloaded per Station: ${got:-none} GiB (want $OFF)"
    python3 -c "import sys; sys.exit(0 if abs(float('${got:-0}')-$OFF) < 1 else 1)" \
      || { log "offload mismatch: placement rule not met, STOP"; bash $HOME/hy4-e3/repo/launch/hy4/hy4-down.sh; echo GB_END; exit 1; }
    grep -h "GPU KV cache size\|Available KV cache\|Model loading took" $OUT/server-rank0.log | tee $OUT/memory.txt
    bench gb vllm glm53 frozen-glm53 check speed speed-heldout prefill needle quality
    bash $HOME/hy4-e3/repo/launch/hy4/hy4-down.sh | tee $OUT/down.txt; echo GB_END ;;
  kday)
    CTX=163840 SEQS=16 OFFGB=166.5 EXTRA_ARGS="--max-logprobs 64 --enable-prompt-tokens-details" \
      bash $HOME/hy4-e3/repo/launch/hy4/hy4-up.sh B $OUT > $OUT/up.log 2>&1 || { tail -20 $OUT/up.log; log "K-day load FAILED"; bash $HOME/hy4-e3/repo/launch/hy4/hy4-down.sh; echo KDAY_END; exit 1; }
    nccl_ib $OUT || { bash $HOME/hy4-e3/repo/launch/hy4/hy4-down.sh; echo KDAY_END; exit 1; }
    bench kday vllm hy4 frozen check speed
    bash $HOME/hy4-e3/repo/launch/hy4/hy4-down.sh | tee $OUT/down.txt; echo KDAY_END ;;
  gnt|gnc|gnc64)  # gnt: Amendment 1 Part 2, G-N thinking-on MMLU-Pro (500, max). gnc: Amendment 2 Part 3, G-N AtCoder code
    bash $L/glm53n-up.sh $OUT > $OUT/up.log 2>&1 || { tail -20 $OUT/up.log; log "G-N load FAILED"; bash $L/glm53n-down.sh; echo ${PHASE^^}_END; exit 1; }
    nccl_ib $OUT || { bash $L/glm53n-down.sh; echo ${PHASE^^}_END; exit 1; }
    if [ $PHASE = gnt ]; then bench gnt sglang glm-5.3 frozen-glm53 check think; else bench $PHASE sglang glm-5.3 frozen-glm53 check code; fi
    bash $L/glm53n-down.sh | tee $OUT/down.txt; echo ${PHASE^^}_END ;;
  hat|hac|hac64)  # H-A (L6-4h stage-3 A recipe, rowmap-hy4-v1-mxfp4, MXFP4). hat: Part 2 MMLU-Pro think. hac: Part 3 code
    E=$HOME/hy4-e3/repo/launch/hy4; RM=rowmap-hy4-v1-mxfp4.json; WF=mxfp4
    E3ROWMAP=$RM WFMT=$WF bash $E/hy4-sidecar.sh 0 > $OUT/sidecar0.log 2>&1 &
    ssh -o BatchMode=yes $R "E3ROWMAP=$RM WFMT=$WF bash $E/hy4-sidecar.sh 1" > $OUT/sidecar1.log 2>&1 &
    wait
    grep -q "^serving" $HOME/hy4-e3/repo/logs/sidecar-rank0.log && ssh -o BatchMode=yes $R "grep -q '^serving' $HOME/hy4-e3/repo/logs/sidecar-rank1.log" \
      || { log "sidecars not serving"; bash $E/hy4-down.sh sidecars; echo ${PHASE^^}_END; exit 1; }
    E3ROWMAP=$RM CTX=163840 SEQS=16 OFFGB=166.5 EXTRA_ARGS="--max-logprobs 64 --enable-prompt-tokens-details" \
      bash $E/hy4-up.sh A $OUT > $OUT/up.log 2>&1 || { tail -20 $OUT/up.log; log "H-A load FAILED"; bash $E/hy4-down.sh sidecars; echo ${PHASE^^}_END; exit 1; }
    tail -4 $OUT/up.log; nccl_ib $OUT || { bash $E/hy4-down.sh sidecars; echo ${PHASE^^}_END; exit 1; }
    if [ $PHASE = hat ]; then bench hat vllm hy4 frozen check think; else bench $PHASE vllm hy4 frozen check code; fi
    bash $E/hy4-down.sh sidecars | tee $OUT/down.txt; echo ${PHASE^^}_END ;;
esac
