#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# launch-hy4-single.sh -- STRETCH, NOT RUNNABLE AS IS: Hy4-FP8 on ONE Station (TP1, no EP): GB300 hot layers + 6000 warm
# experts + Grace cold layers. Written so the layout is concrete; it refuses to start unless FORCE_SINGLE=1.
#
# Why it refuses (tools/hy4_budget.py, 2026-11-20): with TP1 the GB300 holds 18 of 77 MoE layers (235 of 238 GiB at
# util 0.95), so 59 layers (531 GiB of FP8 experts) must sit in Grace. Grace has ~439 GiB MemAvailable. A's mechanism keeps
# a full copy of every warm expert on the GB300 side (masked, for the fallback and the cosine gate), so the 6000 frees
# nothing: 561 GiB > 439 GiB -> OOM while vLLM pins the offloaded tensors. The layout fits only with an EXCLUSIVE warm
# tier (warm experts dropped from the Grace tensors: 394 of 439 GiB, ~45 GiB margin), which needs a loader that allocates
# compact per-layer cold tensors and remaps the expert map (not implemented; see README "Single Station").
# The hook side already works without EP (E3_ALLOW_NO_EP=1 installs an identity expert map; CPU-tested only).
set -euo pipefail
[ "${FORCE_SINGLE:-0}" = 1 ] || { sed -n '3,13p' "$0"; exit 6; }
N=${NAME:-hy4-single}; IMAGE=${IMAGE:-vllm/vllm-openai:hy4}; MODEL=${MODEL:-/models/Hy4-FP8}
OFFGB=${OFFGB:-531.0}; CTX=${CTX:-32768}; SEQS=${SEQS:-16}; MBT=${MBT:-8192}; UTIL=${UTIL:-0.95}; SPEC=${SPEC:-5}
E3DIR=${E3DIR:-$HOME/hy4-e3/repo}; E3ROWMAP=${E3ROWMAP:?single-Station rowmap (build_rowmap.py --ep-size 1)}
E3_SHM=${E3_SHM:-/dev/shm/e3_hy4_peer}; PORT=${PORT:-8021}
LOGDIR=${LOGDIR:-$HOME/bench/hy4-single-$(date +%Y%m%d-%H%M)}; mkdir -p "$LOGDIR"
CACHE=${CACHE:-$HOME/hy4-e3/cache}; mkdir -p "$CACHE"/{vllm,triton,fi}
GB300=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " '/GB300/{print $1;exit}')
SPECCFG="{\"num_speculative_tokens\":$SPEC,\"method\":\"dfly\"}"
docker rm -f "$N" 2>/dev/null || true
docker run -d --name "$N" --device "nvidia.com/gpu=$GB300" --ipc host --network host --ulimit memlock=-1 \
  --cap-add IPC_LOCK -v "$MODEL":/model:ro -v "$CACHE/vllm":/root/.cache/vllm -v "$CACHE/triton":/root/.triton \
  -v "$CACHE/fi":/root/.cache/flashinfer -v "$E3DIR:/e3:ro" -v "$LOGDIR:/e3log" \
  -e PYTHONPATH=/e3/hook -e E3_HOOK=/e3/hook/e3_hook.py -e E3_CONFIG=/model/config.json -e E3_MODE=serve \
  -e E3_ALLOW_NO_EP=1 -e E3_ROWMAP=/e3/rowmaps/$E3ROWMAP -e E3_SHM="$E3_SHM" -e E3_STATS=/e3log/e3-stats-rank0.json \
  -e PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024 -e HF_HUB_OFFLINE=1 \
  --entrypoint vllm "$IMAGE" serve /model --served-model-name hy4 --dtype bfloat16 \
  --reasoning-parser hy_v4 --tool-call-parser hy_v4 --enable-auto-tool-choice --attention-backend FLASHMLA_SPARSE \
  --tensor-parallel-size 1 --moe-backend triton --offload-backend uva --cpu-offload-gb "$OFFGB" \
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight --speculative-config "$SPECCFG" \
  --max-model-len "$CTX" --max-num-seqs "$SEQS" --max-num-batched-tokens "$MBT" --gpu-memory-utilization "$UTIL" \
  --host 127.0.0.1 --port "$PORT" >/dev/null
nohup docker logs -f "$N" > "$LOGDIR/server.log" 2>&1 &
echo "log $LOGDIR/server.log (sidecar: launch/hy4/hy4-sidecar.sh 0 with EP_SIZE=1 rowmap)"
