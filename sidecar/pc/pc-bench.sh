#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# pc-bench.sh LOGDIR SEEDBASE STAGES -- PC: C1 benches with a counter snapshot (flush + both ranks' E3_COUNT dumps) around
# every run. STAGES: real (realbench2 C1 n=3, offset 0, 3 reps = R3.1's held-out ACS bench), catid (8K/1K random C1, NP=3,
# own seed per rep, 3 reps), fit (realbench2 C16 n=33 offset 3 = the 33 bench prompts NOT used by real C1; fit traffic).
# Hygiene: warm-up with its own seed and shape; per-run prefix-cache hit from /metrics deltas (> 2% = CONTAMINATED).
set -uo pipefail
LOGDIR=${1:?}; SB=${2:?}; STAGES=${3:-real catid fit}; P=8010; M=deepseek-v4-pro; R=exx@192.168.101.2; N=dsv4pro-e3
O=$LOGDIR/bench; SN=$LOGDIR/snaps; mkdir -p "$O" "$SN"; S="$O/summary.txt"; K=$HOME/pc-1009/repo/pc
met() { curl -s http://127.0.0.1:$P/metrics | awk '/^vllm:prefix_cache_(hits|queries)_total/{split($1,a,"{"); v[a[1]]+=$NF} END{printf "%d %d\n", v["vllm:prefix_cache_hits_total"], v["vllm:prefix_cache_queries_total"]}'; }
hit() { python3 -c "h0,q0,h1,q1=map(float,'$1 $2'.split()); dq=q1-q0; r=(h1-h0)/dq if dq>0 else 0; print(f'{100*r:.2f}% ({int(h1-h0)}/{int(dq)} tok)' + (' CONTAMINATED' if r>0.02 else ''))"; }
snap() {  # snap NAME
  local f; f=$(python3 "$K/flush.py" http://127.0.0.1:$P); sleep 8
  cp "$LOGDIR/e3-counts-rank0.json" "$SN/$1-r0.json"; ssh -o BatchMode=yes $R "cat $LOGDIR/e3-counts-rank1.json" > "$SN/$1-r1.json"
  echo "$1 $f" >> "$SN/index.txt"
}
B=(docker exec "$N" vllm bench serve --backend vllm --base-url http://127.0.0.1:$P --model $M --tokenizer /model --trust-remote-code)
"${B[@]}" --dataset-name random --random-input-len 512 --random-output-len 64 --num-prompts 4 --max-concurrency 4 --ignore-eos \
  --seed $((SB + 900)) > "$O/warmup.log" 2>&1
echo "== $(date '+%F %T') seedbase $SB stages: $STAGES" | tee -a "$S"
snap s00-start
for ST in $STAGES; do
  for RP in 1 2 3; do
    m0=$(met)
    case $ST in
      real) python3 "$HOME/r31/realbench2.py" http://127.0.0.1:$P $M "$O/real-c1-r$RP.json" 1 3 > "$O/real-c1-r$RP.log" 2>&1
            m1=$(met); echo "real C1 rep$RP: $(tail -1 "$O/real-c1-r$RP.log") hit $(hit "$m0" "$m1")" | tee -a "$S" ;;
      catid) SEED=$((SB + 10 + RP))
            "${B[@]}" --dataset-name random --random-input-len 8192 --random-output-len 1024 --num-prompts 3 --max-concurrency 1 \
              --ignore-eos --seed $SEED --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90 > "$O/catid-c1-r$RP.log" 2>&1
            m1=$(met); T=$(awk '/Mean TPOT/{print $NF}' "$O/catid-c1-r$RP.log"); F=$(awk '/Failed requests/{print $NF}' "$O/catid-c1-r$RP.log")
            echo "catid C1 rep$RP seed $SEED: decode_agg $(python3 -c "print(round(1000/float('${T:-0}' or 1e9),2))") (TPOT $T ms) failed ${F:-?} hit $(hit "$m0" "$m1")" | tee -a "$S" ;;
      fit)  [ $RP = 1 ] || continue
            python3 "$HOME/r31/realbench2.py" http://127.0.0.1:$P $M "$O/fit-c16.json" 16 33 3 > "$O/fit-c16.log" 2>&1
            m1=$(met); echo "fit real C16 n=33 off3: $(tail -1 "$O/fit-c16.log") hit $(hit "$m0" "$m1")" | tee -a "$S" ;;
    esac
    snap "$ST-r$RP"
  done
done
echo "DONE $(date +%T)" | tee -a "$S"
