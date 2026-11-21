#!/usr/bin/env bash
# L6-4h gate G0 (prereg "Stops"): CUDA 13.0 build correctness before any stage-1 measurement.
# Usage: bash ~/l6-4h/g0.sh left|right     Log: ~/l6-4h/g0/<station>.log, ends G0_PASS or G0_FAIL.
set -uo pipefail
ST=$1; W=$HOME/l6-4h; O=$W/g0; mkdir -p "$O"
B30=$W/llama.cpp-cu130/build/bin; B13=$HOME/hy4/llama.cpp/build/bin
STQ=/models/Hy4-GGUF/Hy4-STQ1_0.gguf; WIKI=$W/wiki.test.raw
log() { echo "$(date '+%F %T') $*"; }

# 1. wait for the cu130 build (written by build-cu130.sh)
until grep -q BUILD_CU130_OK "$W/build-cu130.log" 2>/dev/null; do
  pgrep -f 'bash build-cu130.sh' >/dev/null || { sleep 5; grep -q BUILD_CU130_OK "$W/build-cu130.log" || { log "build script exited without BUILD_CU130_OK"; tail -20 "$W/make-cu130.log"; echo G0_FAIL; exit 1; }; }
  sleep 60
done
log "build ok"
"$B30/llama-server" --version > "$O/version-cu130.txt" 2>&1; "$B13/llama-server" --version > "$O/version-cu132.txt" 2>&1
{ ldd "$B30/libggml-cuda.so"; ldd "$B30/llama-server"; ldd "$B30/test-backend-ops"; } > "$O/ldd-cu130.txt"; strings "$B30/libggml-cuda.so" | grep -m3 -iE 'cuda compilation|V13\.' > "$O/nvcc-cu130.txt" || true
grep -q 'not found' "$O/ldd-cu130.txt" && { log "missing libs"; cat "$O/ldd-cu130.txt"; echo G0_FAIL; exit 1; }

uuid() { nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' -v n="$1" '$2 ~ n {print $1}'; }
GB300=$(uuid GB300); RTX=$(uuid 'RTX PRO 6000')
for u in $GB300 $RTX; do nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" && { log "STOP $u busy"; echo G0_FAIL; exit 3; }; done

# 2. test-backend-ops MUL_MAT / MUL_MAT_ID vs CPU on each GPU arch (sm_103a GB300, sm_120a RTX)
TYPES='q4_K|q5_K|q6_K|q8_0|iq2_xxs|iq2_xs|iq3_xxs|iq4_xs|stq1_0|bf16|f16'
fail=0
for g in GB300 RTX; do u=$([ $g = GB300 ] && echo $GB300 || echo $RTX)
  for op in MUL_MAT MUL_MAT_ID; do
    f=$O/tbo-$g-$op.log; log "test-backend-ops $op on $g"
    CUDA_VISIBLE_DEVICES=$u timeout 3600 "$B30/test-backend-ops" test -o $op -b CUDA0 > "$f" 2>&1; rc=$?
    n_ok=$(grep -cE "type_a=($TYPES),.*OK" "$f"); n_bad=$(grep -E "type_a=($TYPES)," "$f" | grep -cvE 'OK|not supported')
    seen=$(grep -oE "type_a=($TYPES)," "$f" | sort -u | tr -d '\n')
    log "$g $op rc=$rc listed-type OK=$n_ok FAIL=$n_bad types=[$seen] tail: $(tail -1 "$f")"
    [ $rc = 0 ] && [ "$n_bad" = 0 ] || fail=1
    grep -q 'type_a=stq1_0,' "$f" || log "WARN stq1_0 not exercised by $op on $g"
  done
done

# 3. left only: STQ1_0 canary + short wikitext-2 PPL, 13.0 build vs 13.2 build, same flags as stage-1 S1
if [ "$ST" = left ]; then
  for b in cu132 cu130; do B=$([ $b = cu130 ] && echo $B30 || echo $B13)
    log "canary $b"
    CUDA_VISIBLE_DEVICES=$GB300 timeout 3600 "$B/llama-completion" -m "$STQ" -ngl 999 -sm none -mg 0 -fa on -lm none \
      -c 4096 --temp 0 --seed 1 -n 64 -no-cnv -p "The capital of France is Paris. The capital of Japan is" \
      > "$O/canary-$b.txt" 2> "$O/canary-$b.err"
    log "ppl $b"
    CUDA_VISIBLE_DEVICES=$GB300 timeout 3600 "$B/llama-perplexity" -m "$STQ" -ngl 999 -sm none -mg 0 -fa on -lm none \
      -f "$WIKI" -c 512 --chunks 64 -b 2048 > "$O/ppl-$b.log" 2>&1
    log "$b PPL: $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' "$O/ppl-$b.log")"
  done
  p13=$(grep -oE 'Final estimate: PPL = [0-9.]+' "$O/ppl-cu132.log" | awk '{print $NF}')
  p30=$(grep -oE 'Final estimate: PPL = [0-9.]+' "$O/ppl-cu130.log" | awk '{print $NF}')
  if [ -z "$p13" ] || [ -z "$p30" ]; then log "PPL missing"; fail=1
  else rel=$(python3 -c "print(abs($p30-$p13)/$p13)"); log "PPL cu130=$p30 cu132=$p13 rel_diff=$rel (gate <= 0.01)"
    python3 -c "import sys; sys.exit(0 if $rel <= 0.01 else 1)" || fail=1
    python3 -c "import sys; sys.exit(0 if $p30 < 50 else 1)" || { log "PPL cu130 implausibly high"; fail=1; }
  fi
  cmp -s "$O/canary-cu130.txt" "$O/canary-cu132.txt" && log "canary identical" || log "canary differs (reported, not gated alone):"
  log "cu130: $(tr '\n' ' ' < "$O/canary-cu130.txt" | head -c 400)"; log "cu132: $(tr '\n' ' ' < "$O/canary-cu132.txt" | head -c 400)"
fi
sha256sum "$O"/* > "$W/g0-$ST.sha256"
[ $fail = 0 ] && echo G0_PASS || echo G0_FAIL
