#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# r31-dspark.sh TAG SPEC SEEDBASE [STAGES="gate flips bench gsm"] -- R3.1 step 3/4: boot E3-A16 + DSpark (SPEC json) with the
# per-layer cosine gate armed (E3_CHECK=3, eager prefill T>=1024), then gate -> flips vs tf-ref-e1-c1 -> bench_r31 -> GSM8K-200.
# Env: UTIL (default 0.98: the drafter adds ~20 GiB/rank of HBM), CTX (default 65536). Leaves the server UP (caller tears down).
set -uo pipefail
TAG=${1:?tag}; export SPEC=${2:?spec json}; SB=${3:?seedbase}; STAGES=${4:-gate flips bench gsm}
export UTIL=${UTIL:-0.98} CTX=${CTX:-65536} E3_CHECK=${E3_CHECK:-3} E3_CHECK_MIN_T=1024 E3_COUNT=0
R=exx@192.168.101.2; D=$HOME/bench/dsv4pro-r31-$(date +%Y%m%d-%H%M)-$TAG; mkdir -p "$D"; API=http://127.0.0.1:8010; M=deepseek-v4-pro
EV=$HOME/dsv4pro/prof/eval; T=$HOME/dsv4pro-e3/repo/tools
log() { echo "$(date '+%T') $*" | tee -a "$D/progress.log"; }
spec() { curl -s $API/metrics | awk '/^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total/{split($1,a,"{"); v[a[1]]+=$NF} END{printf "%d %d %d\n", v["vllm:spec_decode_num_drafts_total"], v["vllm:spec_decode_num_draft_tokens_total"], v["vllm:spec_decode_num_accepted_tokens_total"]}'; }
log "boot $TAG spec=$SPEC util=$UTIL ctx=$CTX logdir $D"
bash ~/r31/r31-up.sh "$D" > "$D/up.txt" 2>&1 || { log "BOOT FAILED"; tail -60 "$D/up.txt" | tee -a "$D/progress.log"; exit 1; }
log "$(tail -1 "$D/up.txt")"
grep -h -i "KV cache size\|Available KV\|speculat\|DSpark\|Model loading took" "$D/server-rank0.log" | tail -8 | tee -a "$D/progress.log"
for s in $STAGES; do case $s in
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
  tail -1 "$D/gsm8k-run.log" | tee -a "$D/progress.log"; log "gsm spec counters $s0 -> $(spec)" ;;
esac; done
for r in 0 1; do curl -s -m 5 $API/metrics >/dev/null; done
docker cp dsv4pro-e3:/e3log/. "$D/e3log-r0" 2>/dev/null; cp $HOME/dsv4pro-e3/repo/logs/e3_peer_stats_rank0.json "$D/" 2>/dev/null
log "DONE stages ($STAGES); server left up"
