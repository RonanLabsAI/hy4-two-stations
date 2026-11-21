#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# bench-hy4.sh OUTDIR SEEDBASE [REPS=3] [CONCS="1 16 32"] -- run on station-left against the rank-0 container (hy4-ep2).
# Derived from tools/bench_e3.sh: random 8K in / 1K out (vllm bench serve, ignore-eos), every run its own seed
# (SEEDBASE + 10*C + rep), a warm-up with a separate seed and shape, prefix-cache hit rate from /metrics deltas (a run
# with hit > 2% is CONTAMINATED and must not be reported). Also the dfly draft acceptance per run from the spec-decode
# counters: on DS-V4.1 DSpark, C1 throughput tracked acceptance at r = 0.99 (dsv41 note, 2026-10-07), so report the mean
# of >= 3 reps and compare A vs B only with acceptance alongside.
set -uo pipefail
O=${1:?outdir}; SB=${2:?seedbase}; REPS=${3:-3}; CONCS=${4:-1 16 32}; N=${NAME:-hy4-ep2}; P=${PORT:-8020}; M=${SERVED:-hy4}
mkdir -p "$O"; S="$O/summary.txt"
met() { curl -s http://127.0.0.1:$P/metrics | awk '
  /^vllm:prefix_cache_(hits|queries)_total/ || /^vllm:spec_decode_num_(accepted|draft)_tokens_total/ {split($1,a,"{"); v[a[1]]+=$NF}
  END{printf "%d %d %d %d\n", v["vllm:prefix_cache_hits_total"], v["vllm:prefix_cache_queries_total"],
      v["vllm:spec_decode_num_accepted_tokens_total"], v["vllm:spec_decode_num_draft_tokens_total"]}'; }
delta() { python3 -c "
a=list(map(float,'$1'.split())); b=list(map(float,'$2'.split()))
dq=b[1]-a[1]; r=(b[0]-a[0])/dq if dq>0 else 0; dd=b[3]-a[3]; acc=(b[2]-a[2])/dd if dd>0 else float('nan')
print(f'hit {100*r:.2f}% ({int(b[0]-a[0])}/{int(dq)} tok)' + (' CONTAMINATED' if r>0.02 else '') + f' accept {acc:.3f}')"; }
# Recipe A: a sidecar timeout silently drops warm routes (peer_tier_v4 _wait OK=0 -> no add), so any run whose summed
# peer timeout counter (both ranks; hook dumps every 10 s) moved is INVALID and must not be reported. Set E3LOG=<LOGDIR>.
tmo() { [ -n "${E3LOG:-}" ] || { echo 0; return; }; python3 -c "
import json,sys; t=0
for s in sys.argv[1:]:
    try: t+=json.loads(s)['peer']['timeouts']
    except Exception: print(-1); sys.exit()
print(t)" "$(cat "$E3LOG/e3-stats-rank0.json" 2>/dev/null)" "$(timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=5 exx@192.168.101.2 cat "$E3LOG/e3-stats-rank1.json" 2>/dev/null)"; }
nvidia-smi --query-gpu=timestamp,name,power.draw,utilization.gpu,memory.used --format=csv -lms 2000 > "$O/power-left.csv" & PL=$!
trap 'kill $PL 2>/dev/null' EXIT
B=(docker exec "$N" vllm bench serve --backend vllm --base-url http://127.0.0.1:$P --model "$M" --tokenizer /model)
"${B[@]}" --dataset-name random --random-input-len 512 --random-output-len 64 --num-prompts 4 --max-concurrency 4 --ignore-eos \
  --seed $((SB + 900)) > "$O/warmup.log" 2>&1
echo "== $(date '+%F %T') seedbase $SB concs $CONCS" | tee -a "$S"
for C in $CONCS; do
  NP=$([ "$C" -le 1 ] && echo 3 || echo $((2 * C)))
  for R in $(seq 1 "$REPS"); do
    SEED=$((SB + 10 * C + R)); m0=$(met); t0=$(tmo)
    "${B[@]}" --dataset-name random --random-input-len 8192 --random-output-len 1024 --num-prompts $NP --max-concurrency $C \
      --ignore-eos --seed $SEED --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90 > "$O/rand-c$C-r$R.log" 2>&1
    m1=$(met); T=$(awk '/Mean TPOT/{print $NF}' "$O/rand-c$C-r$R.log"); F=$(awk '/Failed requests/{print $NF}' "$O/rand-c$C-r$R.log")
    [ -n "${E3LOG:-}" ] && sleep 12; t1=$(tmo)
    TM=$([ "$t0" -lt 0 ] || [ "$t1" -lt 0 ] && echo "timeouts UNREADABLE INVALID" || { d=$((t1 - t0)); [ $d -gt 0 ] && echo "timeouts $d INVALID" || echo "timeouts 0"; })
    echo "rand C$C rep$R seed $SEED: decode_agg $(python3 -c "print(round($C*1000/float('${T:-nan}'),2))" 2>/dev/null) (TPOT $T ms) failed ${F:-?} $(delta "$m0" "$m1") $TM" | tee -a "$S"
  done
done
