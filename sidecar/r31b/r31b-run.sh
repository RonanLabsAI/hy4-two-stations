#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# r31b-run.sh TAG SPEC SEEDBASE [STAGES] -- R3.1b on .94: boot E3-A16 + DSpark through r31b/r31b-up.sh (guard knobs from env)
# and run STAGES in order, stopping at the first HANG (repro exit 2). Leaves the server up unless it died (caller tears down,
# rank 0 first). Stages:
#   arrival sweep soak   r31b/repro_hang.py modes (soak: DURATION, default 1800 s)
#   gate flips bench gsm as r31/r31-dspark.sh (cosine gate, tf flips, bench_r31 C1/C16 real+catid x3, GSM8K-200 thinking C16)
#   div                  E3 peer counters on both ranks + log scan (exceptions / FAILFAST / OOM / shm-broadcast stalls)
# Env: UTIL CTX KVBYTES GUARD R31B_* R31C_* RUNPFX (logdir prefix, default r31b) EXEC_TIMEOUT (see launch-r31b.sh), E3_CHECK (default 3).
set -uo pipefail
TAG=${1:?tag}; export SPEC=${2:?spec json}; SB=${3:?seedbase}; STAGES=${4:-arrival sweep div gate flips bench gsm div soak div}
export UTIL=${UTIL:-0.99} CTX=${CTX:-20480} E3_CHECK=${E3_CHECK:-3} E3_CHECK_MIN_T=1024 E3_COUNT=0
R=exx@192.168.101.2; D=$HOME/bench/dsv4pro-${RUNPFX:-r31b}-$(date +%Y%m%d-%H%M)-$TAG; mkdir -p "$D"; API=http://127.0.0.1:8010; M=deepseek-v4-pro
EV=$HOME/dsv4pro/prof/eval; T=$HOME/dsv4pro-e3/repo/tools; K=$HOME/r31b/repo/r31b
log() { echo "$(date '+%T') $*" | tee -a "$D/progress.log"; }
spec() { curl -s -m 10 $API/metrics | awk '/^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total/{split($1,a,"{"); v[a[1]]+=$NF} END{printf "%d %d %d\n", v["vllm:spec_decode_num_drafts_total"], v["vllm:spec_decode_num_draft_tokens_total"], v["vllm:spec_decode_num_accepted_tokens_total"]}'; }
div() {
  local a b; a=$(sudo -n python3 ~/r31/shm.py 2>/dev/null || python3 ~/r31/shm.py); b=$(ssh -o BatchMode=yes $R "sudo -n python3 ~/r31/shm.py 2>/dev/null || python3 ~/r31/shm.py")
  log "div peer r0 $a"; log "div peer r1 $b"
  scp -q $R:$D/server-rank1.log "$D/server-rank1-copy.log" 2>/dev/null
  for f in "$D/server-rank0.log" "$D/server-rank1-copy.log"; do
    log "div $(basename "$f"): exc=$(grep -c 'WorkerProc hit an exception\|WorkerProc exception' "$f") failfast=$(grep -c 'R31B FAILFAST' "$f") \
cuda_oom=$(grep -c 'CUDA_ERROR_OUT_OF_MEMORY' "$f") alloc_oom=$(grep -c 'allocation failed with OOM' "$f") shm_stall=$(grep -c 'No available shared memory broadcast block' "$f")"
  done
  for r in 0 1; do
    s=$([ $r = 0 ] && cat "$D/r31b-stats-rank0.json" 2>/dev/null || ssh -o BatchMode=yes $R "cat $D/r31b-stats-rank1.json 2>/dev/null")
    [ -n "$s" ] && log "div r31b-stats r$r $(python3 -c "import json,sys;d=json.loads(sys.argv[1]);f=d.get('mem_free_min');print('free_min_MiB',None if f is None else f>>20,'free_last_MiB',(d.get('mem_free_last') or 0)>>20,'failfast',d['failfast'],'injected',d['injected'],'dg',d['draft_dg_gemms'],d['draft_dg_s'],'skip',{k:d.get('skip',{}).get(k) for k in ('calls','light','full','checks')})" "$s")"
  done
  [ -s "$D/r31b-stats-rank0.kctl.json" ] && log "div kctl $(cat "$D/r31b-stats-rank0.kctl.json")"
}
rep() { python3 -I $K/repro_hang.py $API $M $EV/gsm8k_test.jsonl "$1" "$D/repro-$1.jsonl" > "$D/repro-$1.log" 2>&1; local rc=$?
        log "repro $1 exit $rc: $(tail -1 "$D/repro-$1.log")"; return $rc; }
log "boot $TAG spec=$SPEC util=$UTIL ctx=$CTX mbt=${MBT:-8192} kvbytes=${KVBYTES:-} guard=${GUARD:-1} dgwarm=${R31B_DRAFT_DG_WARMUP:-relax} inject=${R31B_INJECT:-} kctl=${R31B_KCTL:-0} koff=${R31B_KCTL_KOFF:-1} skipdraft=${R31C_SKIPDRAFT:-0} probemax=${R31C_PROBE_MAX:-256} check=${R31C_CHECK_EVERY:-0} logdir $D"
bash $K/r31b-up.sh "$D" > "$D/up.txt" 2>&1 || { log "BOOT FAILED"; tail -60 "$D/up.txt" | tee -a "$D/progress.log"; exit 1; }
log "$(tail -1 "$D/up.txt")"
grep -h "R31B\|Available KV\|GPU KV cache size\|KV cache memory" "$D/server-rank0.log" | tail -8 | tee -a "$D/progress.log"
for s in $STAGES; do case $s in
arrival|sweep|soak)
  s0=$(spec); rep $s; rc=$?; log "$s spec counters $s0 -> $(spec)"
  [ $rc = 2 ] && { log "HANG/unhealthy in $s -> stop"; div; exit 2; } ;;
div) div ;;
gate)
  python3 $T/cos_gate.py send $API $M $EV/gsm8k_test.jsonl 6 > "$D/gate-send.log" 2>&1; tail -2 "$D/gate-send.log" | tee -a "$D/progress.log"
  scp -q $R:$D/server-rank1.log "$D/server-rank1-copy.log"
  python3 $T/cos_gate.py parse "$D/server-rank0.log" "$D/server-rank1-copy.log" > "$D/gate.txt" 2>&1; tail -6 "$D/gate.txt" | tee -a "$D/progress.log" ;;
flips)
  s0=$(spec); python3 $EV/tf_flips.py tf $API $M $HOME/dsv4pro/prof/tf-ref-e1-c1.json "$D/tf-vs-e1-c1.json" > "$D/flips.log" 2>&1
  tail -4 "$D/flips.log" | tee -a "$D/progress.log"; log "flips spec counters $s0 -> $(spec)" ;;
bench)
  NAME=dsv4pro-e3 bash ~/r31/bench_r31.sh "$D/bench" "$SB" 3 > "$D/bench-run.log" 2>&1; cat "$D/bench/summary.txt" >> "$D/progress.log" ;;
gsm)
  s0=$(spec); python3 $EV/gsm8k200_v4.py $API/v1/chat/completions $M "$D/gsm8k200.jsonl" 16 200 > "$D/gsm8k-run.log" 2>&1
  tail -1 "$D/gsm8k-run.log" | tee -a "$D/progress.log"; log "gsm spec counters $s0 -> $(spec)"
  curl -sf -m 10 $API/health >/dev/null || { log "server unhealthy after gsm -> stop"; div; exit 2; } ;;
esac; done
docker cp dsv4pro-e3:/e3log/. "$D/e3log-r0" 2>/dev/null
log "DONE stages ($STAGES); server left up"
