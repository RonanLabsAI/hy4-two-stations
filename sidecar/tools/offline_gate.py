#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""Offline per-layer cosine gate for the Hy4 warm-expert weight path (CPU; no GPU, no b12x).

For each layer's experts (read from Hy4-FP8 safetensors), compare the routed-expert output of
  ref    what the GB300 computes: block-FP8 W8A8 (E4M3 weights x FP32 128x128 block scale; activations quantised to
         E4M3 per token per 128-group with a dynamic FP32 scale, both before gate/up and before down)
  cand   what the 6000 computes, emulated: weights converted by sidecar/fp8_convert.py to MXFP4 (W4A8-MX) or MXFP8
         (W8A8-MX), activations quantised MXFP8 E4M3 per 1x32 with UE8M0 scale before each GEMM (b12x A8)
  exact  FP32 activations x dequantised FP8 weights (the checkpoint's own numbers; for context)
on the same inputs and routes. Activation: silu(min(gate, L)) * clamp(up, -L, L) with L = swiglu_limit (the formula
vLLM's hy_v4 model uses, see README). Inputs: N(0, 1) per channel times the layer's post_attention_layernorm weight
when given (the MoE input is RMS-normed), else N(0, 1).

Per layer the gate reports cos(cand, ref) of the routed sum (top-k mixture of the loaded experts with random
normalised weights) and per expert, and PASSes when the layer's mixture cos >= --min-cos (default 0.99 for MXFP4:
see README "Numerics"; 0.999 for MXFP8).

  offline_gate.py --files l1.safetensors l40.safetensors --config models/hy4/config.json --fmt mxfp4 [--json out]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hook"))
sys.path.insert(0, os.path.join(ROOT, "sidecar"))
import fp8_convert as fc  # noqa: E402
import geometry  # noqa: E402


def cos(a, b) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-300))


def act(gate, up, limit):
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return torch.nn.functional.silu(gate) * up


def expert_ref(x, W, limit, block):
    """GB300 block-FP8 W8A8 emulation; W holds the FP8 tensors of one expert."""
    w1 = fc.dequant_block_fp8(W["w1"], W["w1s"], block)
    w3 = fc.dequant_block_fp8(W["w3"], W["w3s"], block)
    w2 = fc.dequant_block_fp8(W["w2"], W["w2s"], block)
    xq = fc.quant_act_fp8_group(x, block[1])
    h = act(xq @ w1.T, xq @ w3.T, limit).to(torch.bfloat16).float()
    return fc.quant_act_fp8_group(h, block[1]) @ w2.T


def expert_exact(x, W, limit, block):
    w1 = fc.dequant_block_fp8(W["w1"], W["w1s"], block)
    w3 = fc.dequant_block_fp8(W["w3"], W["w3s"], block)
    w2 = fc.dequant_block_fp8(W["w2"], W["w2s"], block)
    return act(x @ w1.T, x @ w3.T, limit) @ w2.T


def expert_cand(x, C, fmt, limit, inter):
    w13 = fc.dequant(fmt, C["w13"], C["s13"])
    w2 = fc.dequant(fmt, C["w2"], C["s2"])
    xq = fc.quant_act_mxfp8(x)
    gu = xq @ w13.T
    h = act(gu[:, :inter], gu[:, inter:], limit).to(torch.bfloat16).float()
    return fc.quant_act_mxfp8(h) @ w2.T


def layer_tensors(files, geom):
    """{layer: {expert: {w1, w1s, ...}}, ...} and {layer: norm weight} from safetensors files (only experts present)."""
    from safetensors import safe_open

    out, norms = {}, {}
    pat = re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale)$")
    key = {("gate_proj", "weight"): "w1", ("up_proj", "weight"): "w3", ("down_proj", "weight"): "w2",
           ("gate_proj", "weight_scale"): "w1s", ("up_proj", "weight_scale"): "w3s", ("down_proj", "weight_scale"): "w2s"}
    for f in files:
        with safe_open(f, "pt") as st:
            for name in st.keys():
                m = pat.search(name)
                if m:
                    out.setdefault(int(m.group(1)), {}).setdefault(int(m.group(2)), {})[key[(m.group(3), m.group(4))]] = \
                        st.get_tensor(name)
                m2 = re.search(r"layers\.(\d+)\.post_attention_layernorm\.weight$", name)
                if m2:
                    norms[int(m2.group(1))] = st.get_tensor(name).float()
    for layer, ex in out.items():
        for e, W in ex.items():
            if len(W) != 6:
                raise ValueError(f"layer {layer} expert {e}: incomplete tensors {sorted(W)}")
    return out, norms


def gate_layer(layer, experts, geom, fmt, tokens, seed, norm=None, method="mse"):
    g = torch.Generator().manual_seed(seed + layer)
    ids = sorted(experts)
    x = torch.randn(tokens, geom.hidden, generator=g)
    if norm is not None:
        x = x * norm
    k = min(geom.topk, len(ids))
    route = torch.stack([torch.randperm(len(ids), generator=g)[:k] for _ in range(tokens)])
    wts = torch.rand(tokens, k, generator=g)
    wts = wts / wts.sum(1, keepdim=True)
    block = tuple(geom.weight_block)
    per = {}
    ref_sum = torch.zeros(tokens, geom.hidden)
    cand_sum = torch.zeros(tokens, geom.hidden)
    exact_sum = torch.zeros(tokens, geom.hidden)
    for j, e in enumerate(ids):
        W = experts[e]
        C = fc.convert_expert(W["w1"], W["w1s"], W["w3"], W["w3s"], W["w2"], W["w2s"], fmt=fmt, block=block,
                              method=method)
        r = expert_ref(x, W, geom.swiglu_limit, block)
        c = expert_cand(x, C, fmt, geom.swiglu_limit, geom.inter)
        ex = expert_exact(x, W, geom.swiglu_limit, block)
        wj = (wts * (route == j)).sum(1, keepdim=True)
        ref_sum += wj * r
        cand_sum += wj * c
        exact_sum += wj * ex
        w13 = torch.cat([fc.dequant_block_fp8(W["w1"], W["w1s"], block), fc.dequant_block_fp8(W["w3"], W["w3s"], block)])
        per[e] = {"cos_out": round(cos(c, r), 6), "cos_out_ref_vs_exact": round(cos(r, ex), 6),
                  "cos_w13": round(cos(fc.dequant(fmt, C["w13"], C["s13"]), w13), 6)}
    res = {"layer": layer, "fmt": fmt, "experts": ids, "tokens": tokens, "top_k": k,
           "cos_mix_cand_vs_ref": round(cos(cand_sum, ref_sum), 6),
           "cos_mix_ref_vs_exact": round(cos(ref_sum, exact_sum), 6),
           "cos_mix_cand_vs_exact": round(cos(cand_sum, exact_sum), 6),
           "rel_mix_cand_vs_ref": round(float((cand_sum - ref_sum).norm() / ref_sum.norm()), 5),
           "cos_expert_min": min(v["cos_out"] for v in per.values()), "per_expert": per}
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--config", default=os.path.join(ROOT, "models", "hy4", "config.json"))
    p.add_argument("--fmt", choices=("mxfp4", "mxfp8"), default="mxfp4")
    p.add_argument("--method", choices=("mse", "ocp", "ceil"), default="mse")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-cos", type=float, default=None)
    p.add_argument("--json", default="")
    a = p.parse_args(argv)
    geom = geometry.load(a.config)
    min_cos = a.min_cos if a.min_cos is not None else (0.99 if a.fmt == "mxfp4" else 0.999)
    layers, norms = layer_tensors(a.files, geom)
    results = []
    for layer in sorted(layers):
        r = gate_layer(layer, layers[layer], geom, a.fmt, a.tokens, a.seed, norms.get(layer), a.method)
        r["pass"] = r["cos_mix_cand_vs_ref"] >= min_cos
        results.append(r)
        print(f"layer {layer:2d} {a.fmt}/{a.method}: mix cos(cand, ref) {r['cos_mix_cand_vs_ref']:.6f} "
              f"rel {r['rel_mix_cand_vs_ref']:.4f} | ref vs exact {r['cos_mix_ref_vs_exact']:.6f} | "
              f"expert min {r['cos_expert_min']:.6f} | {'PASS' if r['pass'] else 'FAIL'} (>= {min_cos})", flush=True)
    summary = {"fmt": a.fmt, "method": a.method, "min_cos": min_cos, "layers": [r["layer"] for r in results],
               "PASS": all(r["pass"] for r in results), "results": results}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("fmt", "method", "min_cos", "layers", "PASS")}))
    return 0 if summary["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
