#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""Fetch a few Hy4-FP8 expert tensors (plus router gate) by HTTP range from the pinned HF revision into one small
safetensors file: the CPU fixture for tests/test_hy4_convert.py (HY4_EXPERTS) and tools/offline_gate.py.

  fetch_experts.py model.safetensors.index.json out.safetensors LAYERS EXPERTS [gate.weight,...]
  e.g. fetch_experts.py index.json l20.safetensors 20 0,1,2,3,4,5,6,7 gate.weight   (~305 MB per 8 experts)
"""
import json, struct, sys, urllib.request
from safetensors.torch import save_file
import torch
B = "https://huggingface.co/tencent/Hy4-FP8/resolve/d063b5853ecf56e0776266dab28305fc17a137cc/"
idx = json.load(open(sys.argv[1]))["weight_map"]
out = sys.argv[2]; layers = [int(x) for x in sys.argv[3].split(",")]; experts = [int(x) for x in sys.argv[4].split(",")]
extra = sys.argv[5].split(",") if len(sys.argv) > 5 else []
hdrs = {}
def rng(f, a, b):
    import time
    for i in range(8):
        try:
            req = urllib.request.Request(B + f, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = r.read()
            if len(d) == b - a + 1:
                return d
        except Exception as exc:
            print("retry", i, exc, flush=True)
        time.sleep(2 * (i + 1))
    raise RuntimeError("range fetch failed")
def header(f):
    if f not in hdrs:
        n = struct.unpack("<Q", rng(f, 0, 7))[0]
        hdrs[f] = (8 + n, json.loads(rng(f, 8, 7 + n)))
    return hdrs[f]
DT = {"F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32, "BF16": torch.bfloat16}
names = [f"model.layers.{l}.mlp.experts.{e}.{p}.{s}" for l in layers for e in experts
         for p in ("gate_proj", "up_proj", "down_proj") for s in ("weight", "weight_scale")]
names += [f"model.layers.{l}.mlp.{x}" for l in layers for x in extra]
T = {}
for n in names:
    f = idx[n]; base, h = header(f); m = h[n]; a, b = m["data_offsets"]
    data = rng(f, base + a, base + b - 1)
    T[n] = torch.frombuffer(bytearray(data), dtype=DT[m["dtype"]]).view(m["shape"]).clone()
    print(n, m["dtype"], m["shape"], flush=True)
save_file(T, out)
