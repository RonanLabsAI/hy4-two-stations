#!/usr/bin/env bash
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# e3-down.sh [sidecars] -- run on .94: stop rank 0 then rank 1 (NAME=dsv4pro-e3 ~/stop-dsv4pro.sh), report GB300 MiB.
# With "sidecars", also remove both sidecar containers and their shared buffers.
R=exx@192.168.101.2
NAME=dsv4pro-e3 bash ~/stop-dsv4pro.sh
ssh -o BatchMode=yes $R "NAME=dsv4pro-e3 bash ~/stop-dsv4pro.sh"
docker rm dsv4pro-e3 >/dev/null 2>&1; ssh -o BatchMode=yes $R "docker rm dsv4pro-e3 >/dev/null 2>&1"
if [ "${1:-}" = sidecars ]; then
  docker rm -f dsv4pro-e3-sidecar >/dev/null 2>&1; rm -f /dev/shm/e3_dsv4pro_peer
  ssh -o BatchMode=yes $R "docker rm -f dsv4pro-e3-sidecar >/dev/null 2>&1; rm -f /dev/shm/e3_dsv4pro_peer"
fi
for h in local $R; do
  c="nvidia-smi --query-gpu=name,memory.used --format=csv,noheader"
  if [ $h = local ]; then echo "$(hostname): $($c | tr '\n' ';')"; else echo "right: $(ssh -o BatchMode=yes $h "$c" | tr '\n' ';')"; fi
done
