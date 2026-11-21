#!/usr/bin/env bash
# L6-4h stage 1, station-right, rung Q4_K_M (438 GiB), GB300 + RTX PRO 6000 + Grace overflow. DRAFT, NOT RUN.
# The file on the Station keeps the repo's name: Hy4-Q4_K_M.gguF (capital F).
# Placement (hy4model S3): routed experts of layers 1-23 in Grace, computed on the CPU (--n-cpu-moe 24 covers
# layers 0-23; layer 0 is dense and has none); layers 24-40 experts on the RTX (-ot CUDA1); everything else and
# KV on the GB300. Stock llama.cpp: no C2C zero-copy patch in this tree (K3 receipt: +52% C1 with it; not used).
# Predicted free: GB300 ~3.4 GiB, RTX ~1.7 GiB. Fallback ladder as left-iq2.sh: move one layer down a tier.
RUNG=q4_k_m STATION=right
GGUF=/models/Hy4-GGUF/Hy4-Q4_K_M.gguF
GGUF_SHA=e3903003d394dcce69632da132c946a2515b502780b741b3aaf9e806a4c32dfa
source "$(dirname "$0")/common.sh"
VISIBLE=$GB300          # RTX via rpc_start (one GPU per CUDA process)
preflight "$GB300" "$RTX"; sampler_start; rpc_start
serve --rpc "$RPC_EP" -dev CUDA0,RPC0 -ngl 999 -sm layer -ts 1,0 -mg 0 --n-cpu-moe 24 \
  -ot "blk\.(2[4-9]|3[0-9]|40)\.ffn_(gate|up|down)_exps\.weight=$RPC_BUFT"
measure --think-cap 8192
