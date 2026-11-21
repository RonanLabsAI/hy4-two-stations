#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# r31-nsys.sh TAG [SPEC] -- R3.1 step 2: boot E3-A16 (optionally + SPEC) under nsys on both ranks, warm up (catid 512/64,
# own seed), /start_profile, ONE real-text C1 request (realbench2, held-out ACS prompt 0, 1,024 forced tokens), capture
# 24 iterations (1 prefill + ~23 decode), nsys stop-shutdown, fetch rank-1 report, teardown rank 0 first.
set -uo pipefail
TAG=${1:?tag}; export SPEC=${2:-}
R=exx@192.168.101.2; LOGDIR=$HOME/bench/dsv4pro-r31-$(date +%Y%m%d-%H%M)-$TAG; O=$HOME/r31/out; mkdir -p "$LOGDIR" "$O"
API=http://127.0.0.1:8010
export NSYS_TAG=$TAG EXTRA_ARGS='--profiler-config.profiler=cuda --profiler-config.max_iterations=24'
bash ~/r31/r31-up.sh "$LOGDIR" > "$LOGDIR/up.txt" 2>&1 || { echo "BOOT FAILED"; tail -40 "$LOGDIR/up.txt"; exit 1; }
tail -1 "$LOGDIR/up.txt"
docker exec dsv4pro-e3 vllm bench serve --backend vllm --base-url $API --model deepseek-v4-pro --tokenizer /model \
  --trust-remote-code --dataset-name random --random-input-len 512 --random-output-len 64 --num-prompts 2 --max-concurrency 1 \
  --ignore-eos --seed 7101 > "$LOGDIR/warmup.log" 2>&1
python3 ~/r31/realbench2.py $API deepseek-v4-pro "$LOGDIR/pre-c1.json" 1 2 > "$LOGDIR/pre-c1.log" 2>&1; cat "$LOGDIR/pre-c1.log"
curl -s -m 120 -X POST $API/start_profile; echo " start_profile $(date +%T)"
python3 ~/r31/realbench2.py $API deepseek-v4-pro "$LOGDIR/prof-c1.json" 1 1 > "$LOGDIR/prof-c1.log" 2>&1; cat "$LOGDIR/prof-c1.log"
t1=$(date +%s)
while docker ps -q -f name='^dsv4pro-e3$' | grep -q . || ssh -n $R "docker ps -q -f name='^dsv4pro-e3\$'" | grep -q .; do
  sleep 5; [ $(( $(date +%s) - t1 )) -gt 900 ] && { echo "capture did not end in 15 min"; break; }
done
grep -h -i "nsys\|Generated\|report" "$LOGDIR"/server-rank0.log | tail -3
bash ~/dsv4pro-e3/repo/launch/e3-down.sh
scp -q $R:r31/out/$TAG-r1.nsys-rep $O/ ; ls -la $O/$TAG-r*.nsys-rep
echo "LOGDIR $LOGDIR"
