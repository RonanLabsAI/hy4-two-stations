#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# hy4-sidecar.sh RANK -- start this Station's RTX PRO 6000 warm-expert sidecar for Hy4 EP rank RANK (0 = station-left,
# 1 = station-right) and wait for "serving". Derived from launch/e3-sidecar.sh.
#
# The sidecar converts this rank's warm experts (rowmap ids in [128*RANK, 128*RANK+128)) from block FP8 to the b12x
# format on the 6000 (WFMT=mxfp4: W4A8-MX, ~19.1 MiB/expert; WFMT=mxfp8: W8A8-MX, ~37.1 MiB/expert) and captures
# one CUDA graph per (layer, row bucket). It talks to the GB300 worker only through the shared buffer, so it runs in
# the image b12x master 2cc7f66a is known to work in (vLLM nightly-af7f9488: torch, triton, cuda-python and
# cutlass-dsl 4.7.1 match b12x's pins), not necessarily the hy4 serving image. b12x comes from a source checkout on
# PYTHONPATH; nothing is installed into the image. The RTX must be empty.
#   env: E3DIR (repo copy, default ~/hy4-e3/repo)  B12X (default ~/dsv4pro-e3/b12x, master 2cc7f66a)  E3ROWMAP  WFMT
#   stop: docker rm -f hy4-e3-sidecar && rm -f /dev/shm/e3_hy4_peer
set -euo pipefail
RANK=${1:?rank 0|1}
IMAGE=${SC_IMAGE:-vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b}
MODEL=${MODEL:-/models/Hy4-FP8}
E3DIR=${E3DIR:-$HOME/hy4-e3/repo}; B12X=${B12X:-$HOME/dsv4pro-e3/b12x}
E3ROWMAP=${E3ROWMAP:-rowmap-hy4-prior-mxfp4.json}; WFMT=${WFMT:-mxfp4}; N=hy4-e3-sidecar
E3_SHM=${E3_SHM:-/dev/shm/e3_hy4_peer}
CACHE=${CACHE:-$HOME/hy4-e3/cache-sidecar}; mkdir -p "$CACHE" "$E3DIR/logs"
case "$WFMT" in mxfp4|mxfp8) ;; *) echo "WFMT must be mxfp4|mxfp8"; exit 2 ;; esac
test -s "$E3DIR/rowmaps/$E3ROWMAP" || { echo "rowmap $E3DIR/rowmaps/$E3ROWMAP missing"; exit 3; }
python3 - "$E3DIR/rowmaps/$E3ROWMAP" "$WFMT" <<'PY' || { echo "rowmap/WFMT mismatch"; exit 3; }
import json, sys
rm = json.load(open(sys.argv[1]))
g = rm.get("geometry", {})
assert g.get("family") == "hy_v4", f"rowmap geometry {g} is not Hy4"
assert g.get("warm_fmt") == sys.argv[2], f"rowmap built for {g.get('warm_fmt')} but WFMT={sys.argv[2]}"
PY
RTX=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " '/RTX PRO 6000/{print $1;exit}')
[ -n "$RTX" ] || { echo "no RTX PRO 6000 visible to nvidia-smi on $(hostname)"; exit 4; }
USED=$(nvidia-smi --id="$RTX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "${USED:-99999}" -lt 2048 ] || { echo "RTX busy (${USED} MiB) on $(hostname)"; exit 4; }
test -f "$B12X/b12x/moe/fused_moe/api.py" || { echo "b12x checkout missing at $B12X"; exit 3; }
git -C "$B12X" log -1 --format=%h | grep -q '^2cc7f66' || echo "WARNING: b12x at $(git -C "$B12X" log -1 --format=%h), validated on 2cc7f66a"
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "sidecar IMAGE missing: $IMAGE"; exit 3; }
rm -f "$E3_SHM" 2>/dev/null || true
docker rm -f "$N" 2>/dev/null || true
LOG="$E3DIR/logs/sidecar-rank$RANK.log"; : > "$LOG"
docker run -d --name "$N" --device "nvidia.com/gpu=$RTX" --ipc host --network host --cap-add SYS_NICE \
  --ulimit memlock=-1 --cap-add IPC_LOCK \
  -v "$MODEL":/model:ro -v "$E3DIR":/e3:ro -v "$B12X":/b12x:ro -v "$E3DIR/logs":/logs -v "$CACHE":/root/.cache \
  -v /usr/bin/numactl:/usr/local/bin/numactl:ro \
  -e PYTHONPATH=/b12x -e E3_HOOK_DIR=/e3/hook -e E3_CONFIG=/model/config.json -e E3_SHM="$E3_SHM" \
  -e E3_SMALL_ROWS=64 -e CUDA_DEVICE_ORDER=PCI_BUS_ID --entrypoint bash "$IMAGE" -c \
  "python3 -c 'import b12x, b12x.moe.fused_moe as fm; print(\"b12x\", b12x.__file__, fm.PackedSourceFormat(\"mxfp8_e8m0_k32\"))' && \
   exec numactl --membind=0 python3 /e3/sidecar/peer_server_v4.py --rowmap /e3/rowmaps/$E3ROWMAP --model /model \
     --ep-rank $RANK --ep-size 2 --mode a8 --wfmt $WFMT --logdir /logs" >/dev/null
nohup docker logs -f "$N" > "$LOG" 2>&1 &
echo "$(date '+%F %T') hy4 sidecar rank $RANK on RTX $RTX ($(hostname)), rowmap $E3ROWMAP, $WFMT; waiting for serving"
for i in $(seq 1 360); do
  grep -q "^serving" "$LOG" 2>/dev/null && { tail -4 "$LOG"; nvidia-smi --id="$RTX" --query-gpu=memory.used --format=csv,noheader; exit 0; }
  docker ps -q -f name="^$N\$" | grep -q . || { echo "sidecar died"; tail -30 "$LOG"; exit 1; }
  sleep 5
done
echo "sidecar not serving after 30 min"; tail -20 "$LOG"; exit 1
