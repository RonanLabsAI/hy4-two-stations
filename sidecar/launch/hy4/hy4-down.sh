#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# hy4-down.sh [sidecars] -- run on station-left: stop rank 0 then rank 1 (multi-node vLLM has stranded HBM before when the
# head outlives a worker), report GPU MiB. With "sidecars", also remove both sidecars and their shared buffers.
R=exx@192.168.101.2
docker stop -t 60 hy4-ep2 >/dev/null 2>&1; docker rm hy4-ep2 >/dev/null 2>&1
ssh -o BatchMode=yes $R "docker stop -t 60 hy4-ep2 >/dev/null 2>&1; docker rm hy4-ep2 >/dev/null 2>&1"
if [ "${1:-}" = sidecars ]; then
  docker rm -f hy4-e3-sidecar >/dev/null 2>&1; rm -f /dev/shm/e3_hy4_peer
  ssh -o BatchMode=yes $R "docker rm -f hy4-e3-sidecar >/dev/null 2>&1; rm -f /dev/shm/e3_hy4_peer"
fi
sleep 5
c="nvidia-smi --query-gpu=name,memory.used --format=csv,noheader"
echo "$(hostname): $($c | tr '\n' ';')"
echo "right: $(ssh -o BatchMode=yes $R "$c" | tr '\n' ';')"
echo "(GB300 should be back to ~22 MiB on both; if HBM stays stranded: STOP and report, no reboot)"
