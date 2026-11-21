#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""Routing counts between two E3_COUNT snapshots (the hook's counters accumulate from boot, including vLLM's dummy
profile run, where all max_num_batched_tokens tokens are identical and pick the same experts, and the eager warm-ups).

  counts_delta.py e3-counts-ready-rank0.json e3-counts-rank0.json delta-rank0.json
The output keeps the "layers" (and "by_bucket") sections in the routing-counts format build_rowmap.py --counts reads.
"""
import json
import sys


def sub(a, b):
    return [y - x for x, y in zip(a, b)]


def delta(before: dict, after: dict) -> dict:
    out = {"format": after.get("format"), "source": "counts_delta", "base_time": before.get("time"),
           "time": after.get("time"), "geometry": after.get("geometry"), "layers": {}, "by_bucket": {}}
    for k, v in after["layers"].items():
        d = sub(before["layers"].get(k, [0] * len(v)), v)
        if min(d) < 0:
            raise ValueError(f"layer {k}: counts went down (snapshots from different boots?)")
        out["layers"][k] = d
    for b, layers in (after.get("by_bucket") or {}).items():
        base = (before.get("by_bucket") or {}).get(b, {})
        out["by_bucket"][b] = {k: sub(base.get(k, [0] * len(v)), v) for k, v in layers.items()}
    out["total_routes"] = int(sum(sum(v) for v in out["layers"].values()))
    return out


if __name__ == "__main__":
    d = delta(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))
    json.dump(d, open(sys.argv[3], "w"))
    print(f"{len(d['layers'])} layers, {d['total_routes']} routes in the delta")
