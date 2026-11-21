# L6-4h stage 1, shared launcher. DRAFT, NOT RUN. Sourced by left-stq1_0.sh, left-iq2.sh, right-q4_k_m.sh.
# Runs on the Station as exx. Binds the API to loopback only. Never touches hy4-dl-* containers, the
# RTX PRO 6000 daily routes, LiteLLM, or another session's GPU work.
set -euo pipefail

: "${L64H_APPROVED:?set L64H_APPROVED=1 only inside a PI-approved window (prereg L6-4h, frozen)}"
BIN=${BIN:-$HOME/l6-4h/llama.cpp-cu130/build/bin}   # CUDA 13.0 build (gate G0); ~/hy4 build is nvcc 13.2.51
HARNESS=${HARNESS:-$HOME/l6-4h/tools/hy4/hy4bench.py}
PORT=${PORT:-18080}
BASE=http://127.0.0.1:$PORT
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=$HOME/l6-4h/runs/${RUNG}-${STATION}-${STAMP}
mkdir -p "$OUT"

gpu_uuid() { nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' -v n="$1" '$2 ~ n {print $1}'; }
GB300=$(gpu_uuid 'GB300'); RTX=$(gpu_uuid 'RTX PRO 6000')
[ -n "$GB300" ] && [ -n "$RTX" ] || { echo "STOP: GPU UUIDs not found"; exit 3; }

preflight() {
  # P1 downloads finished: no hy4-dl-* container running (we only read their state)
  if docker ps --format '{{.Names}}' | grep -q '^hy4-dl-'; then echo "STOP: hy4-dl-* still running"; exit 3; fi
  # P2 the GPUs we use are idle (no compute apps)
  for u in "$@"; do
    if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; then echo "STOP: $u busy"; exit 3; fi
  done
  # P3 file identity (HF LFS SHA-256, frozen in the prereg)
  echo "$GGUF_SHA  $GGUF" | sha256sum -c - | tee "$OUT/sha256.txt"
  # P4 build identity
  "$BIN/llama-server" --version 2>&1 | tee "$OUT/llama-version.txt"
  nvidia-smi -q > "$OUT/nvidia-smi-q.txt"; free -b > "$OUT/free-before.txt"
}

sampler_start() {  # no CUDA context; 2 s clocks/power/memory
  nvidia-smi --query-gpu=timestamp,uuid,clocks.sm,power.draw,temperature.gpu,memory.used,utilization.gpu \
    --format=csv -l 2 > "$OUT/gpu-2s.csv" & echo $! > "$OUT/sampler.pid"
}

serve() {  # args: extra llama-server flags
  env CUDA_VISIBLE_DEVICES="$VISIBLE" "$BIN/llama-server" -m "$GGUF" \
    --host 127.0.0.1 --port "$PORT" --metrics --jinja -sp \
    -fit off -fa on -lm none -np 16 -kvu -c 163840 -b 4096 -ub 2048 -t 64 -tb 72 \
    "$@" > "$OUT/server.log" 2>&1 &
  echo $! > "$OUT/server.pid"
  local t0=$(date +%s)
  until curl -sf "$BASE/health" >/dev/null; do
    sleep 5
    kill -0 "$(cat "$OUT/server.pid")" 2>/dev/null || { echo "STOP: server exited during load"; tail -50 "$OUT/server.log"; exit 4; }
    [ $(( $(date +%s) - t0 )) -lt 3600 ] || { echo "STOP: load > 1 h"; stop; exit 4; }
  done
  echo "load_to_health_s $(( $(date +%s) - t0 ))" | tee "$OUT/load.txt"
}

# One GPU per CUDA process on these Stations (driver R595: a process that sees both the GB300 and the RTX fails
# ggml_cuda_init, "initialization error"; L5 Finding 1, re-seen 2026-11-20). So the RTX runs in its own
# ggml-rpc-server and llama-server (GB300 only) reaches it as RPC0 over loopback. Deviation from the draft's
# in-process CUDA1 placement: same tensors on the RTX, plus a loopback hop per offloaded layer.
RPC_EP=127.0.0.1:50053
RPC_BUFT="RPC0[$RPC_EP]"
rpc_start() {
  env CUDA_VISIBLE_DEVICES="$RTX" "$BIN/ggml-rpc-server" -H 127.0.0.1 -p 50053 > "$OUT/rpc-server.log" 2>&1 &
  echo $! > "$OUT/rpc.pid"
  for _ in $(seq 30); do grep -q 'endpoint' "$OUT/rpc-server.log" && return 0; sleep 1; done
  echo "STOP: rpc-server did not start"; cat "$OUT/rpc-server.log"; exit 4
}

measure() {
  python3 -I "$HARNESS" all --base "$BASE" --engine llamacpp --rung "$RUNG" --out "$OUT/bench" --think "$@" \
    2>&1 | tee "$OUT/bench.log"
}

stop() {
  kill "$(cat "$OUT/server.pid")" 2>/dev/null || true
  kill "$(cat "$OUT/sampler.pid")" 2>/dev/null || true
  # llama-server queries the RPC device while exiting; killing rpc-server first made it abort (Q4_K_M 21:58).
  local sp; sp=$(cat "$OUT/server.pid" 2>/dev/null); for _ in $(seq 60); do kill -0 "$sp" 2>/dev/null || break; sleep 1; done
  [ -f "$OUT/rpc.pid" ] && kill "$(cat "$OUT/rpc.pid")" 2>/dev/null || true
  sleep 10; free -b > "$OUT/free-after.txt"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv > "$OUT/apps-after.csv"
}
trap stop EXIT
