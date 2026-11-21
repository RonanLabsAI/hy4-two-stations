#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""Fetch post_attention_layernorm weights (the MoE input scale) for tools/offline_gate.py inputs.

  fetch_norms.py model.safetensors.index.json norms.safetensors 1,20,40,60,77
"""
import json, struct, sys, urllib.request, time
import torch
from safetensors.torch import save_file
B = "https://huggingface.co/tencent/Hy4-FP8/resolve/d063b5853ecf56e0776266dab28305fc17a137cc/"
idx = json.load(open(sys.argv[1]))["weight_map"]; out = sys.argv[2]; layers = [int(x) for x in sys.argv[3].split(",")]
def rng(f, a, b):
    for i in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(B + f, headers={"Range": f"bytes={a}-{b}"}), timeout=120) as r:
                d = r.read()
            if len(d) == b - a + 1: return d
        except Exception as e: print("retry", e, flush=True)
        time.sleep(2 * (i + 1))
    raise RuntimeError
T = {}
for l in layers:
    n = f"model.layers.{l}.post_attention_layernorm.weight"; f = idx[n]
    hl = struct.unpack("<Q", rng(f, 0, 7))[0]; h = json.loads(rng(f, 8, 7 + hl)); m = h[n]; a, b = m["data_offsets"]
    T[n] = torch.frombuffer(bytearray(rng(f, 8 + hl + a, 8 + hl + b - 1)), dtype=torch.bfloat16).view(m["shape"]).clone()
    print(n, m["shape"], float(T[n].float().abs().mean()), float(T[n].float().abs().max()), flush=True)
save_file(T, out)
