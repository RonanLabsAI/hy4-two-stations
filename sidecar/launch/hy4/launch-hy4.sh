#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# launch-hy4.sh RANK -- Tencent Hy4-FP8 across BOTH GB300 Stations: TP2 + EP2 (vLLM native multi-node, mp backend,
# --nnodes 2) over rail 1 (mlx5_1 / enP1p3s0f1np1, 192.168.101.0/24), routed experts of the first MoE layers UVA-offloaded
# to Grace, dfly drafter on. Derived from launch/launch-dsv4pro-e3.sh (R2.7) with the Hy4 vendor flags.
#
# Recipes (same container, chosen by env):
#   B      E3HOOK=0                     GB300 hot layers + Grace cold layers (stock vLLM; the baseline)
#   COUNT  E3HOOK=1 E3_MODE=count       B plus routing-count capture (hook only counts; no sidecar, no rowmap)
#   A      E3HOOK=1 [E3_MODE=serve]     B plus the 6000 warm tier (launch/hy4/hy4-sidecar.sh RANK must be serving)
# All three use the SAME placement (--cpu-offload-gb OFFGB, layer-granular, layers 1..37 at the default 166.5), the same
# MoE backend (triton: modular, honours expert_map, keeps UVA tensors in place) and the same drafter, so B vs A
# differs only in where the warm experts' routes run.
#
# Vendor launch (TP8): --dtype bfloat16 --reasoning-parser hy_v4 --tool-call-parser hy_v4
#   --speculative-config '{"num_speculative_tokens":5,"method":"dfly"}' --attention-backend FLASHMLA_SPARSE
# Changes for two Stations: TP2 + --enable-expert-parallel, --moe-backend triton (auto would pick FLASHINFER_TRTLLM:
# monolithic, so no expert_map hook point, and its load-time weight shuffle allocates new GPU tensors, undoing UVA),
# --offload-backend uva --cpu-offload-gb OFFGB --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight
# (the param path is ...mlp.experts.routed_experts.w13_weight; "experts.w13_weight" matches nothing).
#
# usage: [OFFGB=166.5] [CTX=65536] [SEQS=32] [MBT=8192] [UTIL=0.95] [LMO=0] [SPEC=5] [DD=1]
#        [E3HOOK=0|1] [E3_MODE=serve|count] [E3ROWMAP=rowmap-hy4-prior-mxfp4.json] [E3_CHECK=0] [E3_COUNT=0|1]
#        [LOGDIR=~/bench/hy4-...] [EXTRA_ARGS="..."] launch-hy4.sh RANK
#   Start RANK 1 on station-right first, then RANK 0 on station-left (launch/hy4/hy4-up.sh does both).
#   API: http://127.0.0.1:${PORT:-8020}/v1 (loopback only, prereg L6-4h)  model ${SERVED:-hy4}
# TEARDOWN: launch/hy4/hy4-down.sh on station-left (rank 0 first, then rank 1), then check both GB300s are ~22 MiB.
set -euo pipefail
RANK=${1:?rank 0|1}; HEAD=192.168.101.1; N=${NAME:-hy4-ep2}
IMAGE=${IMAGE:-vllm/vllm-openai:hy4}; DD_IMAGE=$IMAGE
MODEL=${MODEL:-/models/Hy4-FP8}; SHARDS=${SHARDS:-245}
SERVED=${SERVED:-hy4}; PORT=${PORT:-8020}; MPORT=${MPORT:-29624}
OFFGB=${OFFGB:-166.5}; PIN=${PIN:-alloc}; MOE=${MOE:-triton}
CTX=${CTX:-65536}; SEQS=${SEQS:-32}; MBT=${MBT:-8192}; UTIL=${UTIL:-0.95}; LMO=${LMO:-0}; SPEC=${SPEC:-5}
MINAVAIL=${MINAVAIL:-230}
OPARAMS=${OPARAMS:-routed_experts.w13_weight routed_experts.w2_weight}
LOGDIR=${LOGDIR:-$HOME/bench/hy4-$(date +%Y%m%d-%H%M)}; mkdir -p "$LOGDIR"
CACHE=${CACHE:-$HOME/hy4-e3/cache}; mkdir -p "$CACHE"/{vllm,triton,tilelang,nv,fi,dj} "$HOME/hy4-e3/prof"
MYIP=$([ "$RANK" = 0 ] && echo 192.168.101.1 || echo 192.168.101.2)

# --- preflight -----------------------------------------------------------------------------------------------------------
test -s "$MODEL/model.safetensors.index.json" && test -s "$MODEL/config.json" \
  && [ "$(ls "$MODEL"/model-*-of-*.safetensors 2>/dev/null | wc -l)" = "$SHARDS" ] \
  || { echo "WEIGHTS MISSING/INCOMPLETE on $(hostname): $MODEL (want $SHARDS shards)"; exit 3; }
[ "$(sha256sum "$MODEL/config.json" | cut -c1-16)" = "2ae591dd0bc9a9a0" ] \
  || { echo "config.json is not tencent/Hy4-FP8@d063b585 (sha256 2ae591dd...)"; exit 3; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "IMAGE MISSING on $(hostname): $IMAGE"; exit 3; }
GB300=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " '/GB300/{print $1;exit}')
[ -n "$GB300" ] || { echo "no GB300 visible to nvidia-smi on $(hostname)"; exit 4; }
USED=$(nvidia-smi --id="$GB300" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "${USED:-99999}" -lt 8192 ] || { echo "GB300 busy/stranded (${USED} MiB) on $(hostname)"; exit 4; }
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
AVAIL=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$AVAIL" -ge "$MINAVAIL" ] || { echo "MemAvailable ${AVAIL} GiB < ${MINAVAIL} GiB on $(hostname)"; exit 5; }
grep -E "MemAvailable|Shmem:|Mlocked|Unevictable|AnonPages|Cached:" /proc/meminfo > "$LOGDIR/meminfo-pre-rank$RANK.txt"

# --- O1 fabric block (as launch-dsv4pro-e3.sh): Data Direct overlay + tuner3 + NCCL_IB_TC=106; DD=0 = off -----------------
DD=${DD:-1}; DDARGS=(); TUNER_CSV=${TUNER_CSV:-tuner3.csv}
if [ "$DD" = 1 ]; then
  nm -D "$HOME/hostrdma/libmlx5.so.1" | grep -q mlx5dv_get_data_direct_sysfs_path || { echo "~/hostrdma not staged (DD=0 to skip)"; exit 3; }
  for f in libnccl-tuner-example.so "$TUNER_CSV"; do [ -s "$HOME/nccl-tuner/$f" ] || { echo "~/nccl-tuner/$f missing"; exit 3; }; done
  _C=$(docker create "$DD_IMAGE" true); _T=$(mktemp -d)
  docker cp "$_C:/usr/lib/aarch64-linux-gnu/libibverbs.so.1" "$_T/" >/dev/null; docker cp "$_C:/usr/lib/aarch64-linux-gnu/libmlx5.so.1" "$_T/" >/dev/null
  docker rm "$_C" >/dev/null; _IBV=$(readlink "$_T/libibverbs.so.1"); _MLX=$(readlink "$_T/libmlx5.so.1"); rm -rf "$_T"
  [ -n "$_IBV" ] && [ -n "$_MLX" ] || { echo "could not resolve image rdma lib names (DD=0 to skip)"; exit 3; }
  [ -c /dev/infiniband/rdma_cm ] && DDARGS+=(--device /dev/infiniband/rdma_cm)
  DDARGS+=(-v "$HOME/hostrdma/libibverbs.so.1:/usr/lib/aarch64-linux-gnu/$_IBV:ro"
           -v "$HOME/hostrdma/libmlx5.so.1:/usr/lib/aarch64-linux-gnu/$_MLX:ro"
           -v /usr/lib/aarch64-linux-gnu/libibverbs:/usr/lib/aarch64-linux-gnu/libibverbs:ro
           -v /etc/libibverbs.d:/etc/libibverbs.d:ro
           -v "$HOME/nccl-tuner:/opt/nccl-tuner:ro"
           -e NCCL_TUNER_PLUGIN=/opt/nccl-tuner/libnccl-tuner-example.so -e NCCL_TUNER_CONFIG_FILE=/opt/nccl-tuner/$TUNER_CSV
           -e NCCL_IB_QPS_PER_CONNECTION=4 -e NCCL_IB_SPLIT_DATA_ON_QPS=1 -e NCCL_IB_TC=106)
fi

# --- engine env / args ---------------------------------------------------------------------------------------------------
NCCL=(-e NCCL_SOCKET_IFNAME=enP1p3s0f1np1 -e GLOO_SOCKET_IFNAME=enP1p3s0f1np1 -e TP_SOCKET_IFNAME=enP1p3s0f1np1
      -e NCCL_IB_HCA==mlx5_1:1 -e NCCL_IB_DISABLE=0 -e NCCL_NET_GDR_LEVEL=SYS -e NCCL_DMABUF_ENABLE=1
      -e NCCL_MNNVL_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET)
[ -n "${NCCL_PROTO:-}" ] && NCCL+=(-e NCCL_PROTO="$NCCL_PROTO")   # L6-4h K- control (NCCL_PROTO=Simple)
ENVS=(-e VLLM_HOST_IP="$MYIP" -e VLLM_ENGINE_READY_TIMEOUT_S=5400 -e VLLM_LOGGING_LEVEL=INFO
      -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 -e VLLM_ALLREDUCE_USE_FLASHINFER=0 -e VLLM_USE_NCCL_SYMM_MEM=0
      -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)
case "$PIN" in
  alloc) ENVS+=(-e PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024) ;;
  nopin) ENVS+=(-e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1) ;;
  *) echo "PIN must be alloc|nopin"; exit 2 ;;
esac
case "$MOE" in triton|cutlass) ;; *) echo "MOE=$MOE: the hook and UVA offload need a modular backend that keeps the"\
  " offloaded tensors (triton; cutlass untested). TRTLLM/DeepGEMM rewrite weights at load."; exit 2 ;; esac
SPECCFG="{\"num_speculative_tokens\":$SPEC,\"method\":\"dfly\"}"

ARGS=(serve /model --served-model-name "$SERVED" --dtype bfloat16
      --reasoning-parser hy_v4 --tool-call-parser hy_v4 --enable-auto-tool-choice
      --attention-backend FLASHMLA_SPARSE
      --distributed-executor-backend mp --nnodes 2 --node-rank "$RANK" --master-addr "$HEAD" --master-port "$MPORT"
      --tensor-parallel-size 2 --pipeline-parallel-size 1 --enable-expert-parallel
      --disable-custom-all-reduce --moe-backend "$MOE"
      --offload-backend uva --cpu-offload-gb "$OFFGB" --cpu-offload-params $OPARAMS
      --max-model-len "$CTX" --max-num-seqs "$SEQS" --max-num-batched-tokens "$MBT" --gpu-memory-utilization "$UTIL")
[ "$SPEC" -gt 0 ] && ARGS+=(--speculative-config "$SPECCFG")
[ "$LMO" = 1 ] && ARGS+=(--language-model-only)
if [ "$RANK" = 0 ]; then ARGS+=(--host 127.0.0.1 --port "$PORT"); else ARGS+=(--headless); fi
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)

MOUNTS=(-v "$MODEL":/model:ro -v "$CACHE/vllm":/root/.cache/vllm -v "$CACHE/triton":/root/.triton
        -v "$CACHE/tilelang":/root/.tilelang -v "$CACHE/nv":/root/.nv -v "$CACHE/fi":/root/.cache/flashinfer
        -v "$CACHE/dj":/root/.dj -v "$HOME/hy4-e3/prof":/prof)

# --- E3 block ------------------------------------------------------------------------------------------------------------
E3HOOK=${E3HOOK:-0}; E3_MODE=${E3_MODE:-serve}; E3DIR=${E3DIR:-$HOME/hy4-e3/repo}
E3ROWMAP=${E3ROWMAP:-rowmap-hy4-prior-mxfp4.json}; E3_SHM=${E3_SHM:-/dev/shm/e3_hy4_peer}
if [ "$E3HOOK" = 1 ]; then
  test -s "$E3DIR/hook/e3_hook.py" && test -s "$E3DIR/hook/geometry.py" || { echo "E3 repo missing under $E3DIR"; exit 3; }
  if [ "$E3_MODE" = serve ]; then
    test -s "$E3DIR/rowmaps/$E3ROWMAP" || { echo "rowmap $E3ROWMAP missing under $E3DIR/rowmaps"; exit 3; }
    grep -q "^serving" "$E3DIR/logs/sidecar-rank$RANK.log" 2>/dev/null && test -e "$E3_SHM" \
      || { echo "E3 sidecar not serving on $(hostname) (run launch/hy4/hy4-sidecar.sh $RANK first)"; exit 4; }
  fi
  MOUNTS+=(-v "$E3DIR:/e3:ro" -v "$LOGDIR:/e3log")
  ENVS+=(-e PYTHONPATH=/e3/hook -e E3_HOOK=/e3/hook/e3_hook.py -e E3_CONFIG=/model/config.json -e E3_MODE="$E3_MODE"
         -e E3_ROWMAP=/e3/rowmaps/$E3ROWMAP -e E3_ENABLE=${E3_ENABLE:-1} -e E3_SHM="$E3_SHM"
         -e E3_CHECK=${E3_CHECK:-0} -e E3_CHECK_MIN_T=${E3_CHECK_MIN_T:-1024} -e E3_STATS=/e3log/e3-stats-rank$RANK.json
         -e E3_SMALL_ROWS=64 -e E3_PEER_SPINS=${E3_PEER_SPINS:-20000000})
  { [ "${E3_COUNT:-0}" = 1 ] || [ "$E3_MODE" = count ]; } && ENVS+=(-e E3_COUNT=/e3log/e3-counts-rank$RANK.json)
fi
echo "e3hook=$E3HOOK mode=$E3_MODE rowmap=$E3ROWMAP check=${E3_CHECK:-0} count=${E3_COUNT:-0} offgb=$OFFGB moe=$MOE spec=$SPEC" > "$LOGDIR/e3-rank$RANK.txt"

docker rm -f "$N" 2>/dev/null || true
docker run -d --name "$N" --device "nvidia.com/gpu=$GB300" --ipc host --network host --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK --cap-add SYS_NICE --device /dev/infiniband/uverbs1 "${DDARGS[@]}" \
  "${MOUNTS[@]}" "${ENVS[@]}" "${NCCL[@]}" \
  --entrypoint vllm "$IMAGE" "${ARGS[@]}" >/dev/null
nohup docker logs -f "$N" > "$LOGDIR/server-rank$RANK.log" 2>&1 &
docker inspect --format '{{.Image}}' "$N" > "$LOGDIR/image-digest-rank$RANK.txt"
{ echo "launched $(date '+%F %T %Z') host=$(hostname) rank=$RANK gpu=$GB300 image=$IMAGE"
  echo "model=$MODEL offgb=$OFFGB moe=$MOE pin=$PIN ctx=$CTX seqs=$SEQS mbt=$MBT util=$UTIL lmo=$LMO spec=$SPEC dd=$DD nccl_proto=${NCCL_PROTO:-default}"
  printf 'args:'; printf ' %q' "${ARGS[@]}"; echo; } | tee "$LOGDIR/launch-rank$RANK.txt"
echo "log $LOGDIR/server-rank$RANK.log"
