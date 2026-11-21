#!/usr/bin/env bash
# L6-4h stage 1, station-left, rung IQ2 (310 GiB), GB300 + RTX PRO 6000, all GPU. DRAFT, NOT RUN. Runs after STQ1_0.
# Placement (hy4model S2): attention, shared experts, head, KV and routed experts of layers 24-77 on the GB300
# (CUDA0); routed experts of layers 1-23 on the RTX (CUDA1) via -ot. -ts 1,0 keeps every whole layer on CUDA0.
# Predicted free: GB300 ~4.1 GiB, RTX ~2.8 GiB. Fallback ladder (pre-registered, max 2 steps, each recorded):
# OOM on CUDA0 -> move layer 24 to the RTX; OOM on CUDA1 -> move layer 23 back and use -c 147456.
RUNG=iq2 STATION=left
GGUF=/models/Hy4-GGUF/Hy4-IQ2.gguf
GGUF_SHA=75884fbc57be7b5cd91d2db9cf9be6cd4745676dcbe4be526a7efe611df942d3
source "$(dirname "$0")/common.sh"
VISIBLE=$GB300          # RTX via rpc_start (one GPU per CUDA process)
preflight "$GB300" "$RTX"; sampler_start; rpc_start
serve --rpc "$RPC_EP" -dev CUDA0,RPC0 -ngl 999 -sm layer -ts 1,0 -mg 0 \
  -ot "blk\.([1-9]|1[0-9]|2[0-3])\.ffn_(gate|up|down)_exps\.weight=$RPC_BUFT"
measure
