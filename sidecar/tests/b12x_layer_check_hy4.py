# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""One-layer numerics check of the Hy4 sidecar's b12x call (RUN ON AN RTX PRO 6000; not part of the CPU suite).

  E3_CONFIG=/model/config.json python b12x_layer_check_hy4.py --model /model --layer 20 --experts 0-35 \
      [--wfmt mxfp4|mxfp8] [--layout w31|w13] [--tokens 64]

Builds this layer's experts exactly as sidecar/peer_server_v4.py does (block FP8 -> fp8_convert -> b12x), runs b12x on
random routes, and compares against
  emu  the same converted weights in plain torch with b12x's A8 activation quantisation emulated (MXFP8 per 32):
       isolates b12x's layout / scale / SwiGLU-clamp contract. PASS needs cos >= 0.999 (the DS-V4 kit's bar).
  ref  the GB300's block-FP8 W8A8 computation emulated (tools/offline_gate.py expert_ref): the end-to-end numerics
       change the warm tier introduces. Expected (offline, CPU, 2026-11-20): mxfp4 ~0.98, mxfp8 ~0.997.
A wrong --layout shows up as emu cos near 0; a scale-convention error as a norm ratio far from 1.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.environ.get("E3_HOOK_DIR", "/e3/hook"))
sys.path.insert(0, "/e3/sidecar")
sys.path.insert(0, "/e3/tools")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("hook", "sidecar", "tools"):
    sys.path.insert(0, os.path.join(HERE, d))
import fp8_convert as fc  # noqa: E402
import offline_gate as og  # noqa: E402
import peer_tier_v4 as pt  # noqa: E402
from peer_server_v4 import load_layer_experts, request_for_capacity, tensor_reader  # noqa: E402

import b12x.moe.fused_moe as fm  # noqa: E402


def parse_ids(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-"))
            out += list(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/model")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--experts", default="0-35")
    p.add_argument("--wfmt", choices=("mxfp4", "mxfp8"), default="mxfp4")
    p.add_argument("--layout", default="w31")
    p.add_argument("--tokens", type=int, default=64)
    a = p.parse_args()
    G = pt.GEOM
    assert G.family == "hy_v4", "set E3_CONFIG=/model/config.json"
    H, I, K, L = G.hidden, G.inter, G.topk, G.swiglu_limit
    dev = torch.device("cuda", 0)
    ids = parse_ids(a.experts)
    n = len(ids)
    tensor = tensor_reader(a.model, raw=False)
    W = load_layer_experts(tensor, a.layer, ids, dev, a.wfmt)
    if a.layout == "w13":   # deliberately swapped halves, to show what a wrong layout looks like
        W["w13"] = torch.cat([W["w13"][:, I:], W["w13"][:, :I]], 1)
        W["s13"] = torch.cat([W["s13"][:, I:], W["s13"][:, :I]], 1)
    g = torch.Generator(device=dev).manual_seed(0)
    T = a.tokens
    x = torch.randn(T, H, device=dev, generator=g).to(torch.bfloat16)
    topk = torch.stack([torch.randperm(n, device=dev, generator=g)[:K] for _ in range(T)]).to(torch.int32)
    w = torch.rand(T, K, device=dev, generator=g)
    w = w / w.sum(1, keepdim=True) * G.routed_scaling

    # torch references
    emu = torch.zeros(T, H, device=dev)
    ref = torch.zeros(T, H, device=dev)
    raw = {e: {k: tensor(v).to(dev) for k, v in G.expert_names(a.layer, e).items()} for e in ids}
    for j, e in enumerate(ids):
        C = {k: W[k][j] for k in W}
        m = (topk == j)
        wj = (w * m).sum(1, keepdim=True)
        if not bool(m.any()):
            continue
        emu += wj * og.expert_cand(x.float(), C, a.wfmt, L, I)
        ref += wj * og.expert_ref(x.float(), raw[e], L, tuple(G.weight_block))

    src = "fp4_e8m0_k32" if a.wfmt == "mxfp4" else "mxfp8_e8m0_k32"
    plan = fm.plan_weights(
        source=fm.PackedSource(format=fm.PackedSourceFormat(src), w13_layout=fm.W13Layout("w31")),
        activation=fm.ActivationSpec(mode=fm.ActivationMode.A8, nonlinearity="silu", io_dtype=torch.bfloat16,
                                     swiglu_limit=L, swiglu_alpha=None, swiglu_beta=None),
        geometry=fm.MoEGeometry(num_experts=n, hidden_size=H, intermediate_size=I))
    ones = torch.ones(n, device=dev)
    one = torch.ones((), device=dev)
    ex = fm.prepare_weights(plan=plan, weights=fm.PackedWeights(
        w13=W["w13"], w2=W["w2"], w13_block_scales=W["s13"].clamp(max=247).view(torch.float8_e8m0fnu),
        w2_block_scales=W["s2"].clamp(max=247).view(torch.float8_e8m0fnu), w13_global_scales=ones,
        w2_global_scales=ones, input_scale=one, intermediate_scale=one, immutable_input_scales=True))
    cap = max(64, T)
    decl = fm.plan_execution(experts=ex, capacity=fm.ExecutionCapacity(
        max_tokens=cap, top_k=K, warmup_token_counts=tuple(m for m in (1, 2, 4, 8, 16, 32, 64, 128, 256)
                                                             if m <= cap) or (cap,), route_num_experts=0))
    out = torch.zeros(cap, H, dtype=torch.bfloat16, device=dev)
    xb = torch.zeros(cap, H, dtype=torch.bfloat16, device=dev)
    xb[:T] = x
    tb = torch.full((cap, K), -1, dtype=torch.int32, device=dev)
    tb[:T] = topk
    tb[T:, 0] = 0
    wb = torch.zeros(cap, K, device=dev)
    wb[:T] = w
    from dataclasses import replace

    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.preparation.types import require_prepared

    def call(st, m):
        sc = tuple(torch.empty(int(sp.shape[0]), dtype=sp.dtype, device=sp.device) for sp in st.scratch.scratch_specs())
        b = st.bind(scratch=sc, a=xb[:m], experts=ex, topk_ids=tb[:m], topk_weights=wb[:m], output=out[:m],
                    input_scales_static=True)
        return PreparedCall(run=b.run, output=out[:m], owners=(b,))
    calls = {m: (lambda st, m=m: call(st, m)) for m in decl.token_counts}
    src_x = xb.clone()

    def race(st, m):
        return replace(call(st, m), produce=lambda: xb[:m].copy_(src_x[:m]))
    PreparationSession(device=dev).prepare((request_for_capacity(
        decl, name="hy4chk", calls=calls, benchmark_calls={m: (lambda st, m=m: race(st, m)) for m in calls}),))
    m = min(c for c in decl.token_counts if c >= T)
    variant = decl.variants[m]
    st = require_prepared(variant, variant.component_id)
    sc = tuple(torch.empty(int(sp.shape[0]), dtype=sp.dtype, device=sp.device) for sp in st.scratch.scratch_specs())
    b = fm.bind(variant, scratch=sc, a=xb[:m], experts=ex, topk_ids=tb[:m], topk_weights=wb[:m], output=out[:m],
                input_scales_static=True)
    fm.run(binding=b)
    torch.cuda.synchronize()
    o = out[:T].float()
    c_emu, c_ref = og.cos(o, emu), og.cos(o, ref)
    print(f"hy4 layer={a.layer} experts={n} wfmt={a.wfmt} layout={a.layout} T={T}: cos(b12x, emu)={c_emu:.6f} "
          f"cos(b12x, fp8 ref)={c_ref:.6f} cos(emu, fp8 ref)={og.cos(emu, ref):.6f} "
          f"|b12x|/|emu|={float(o.norm() / emu.norm()):.4f} finite={bool(torch.isfinite(o).all())}")
    # Gate both: kernel vs its emulation (layout/packing) AND vs the real FP8 checkpoint (catches a layout error that
    # the emulation shares, e.g. --layout w13, which swaps W before emu is built). Ref floors = the format's
    # offline cosine minus margin (README-HY4: MXFP4 0.979-0.987, MXFP8 0.997-0.998).
    min_ref = float(os.environ.get("MIN_COS", 0.97 if a.wfmt == "mxfp4" else 0.995))
    ok = c_emu >= 0.999 and c_ref >= min_ref and bool(torch.isfinite(o).all())
    print(f"gate: cos(b12x, emu)>=0.999 and cos(b12x, fp8 ref)>={min_ref} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
