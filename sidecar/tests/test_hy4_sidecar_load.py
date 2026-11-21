# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""CPU test of the sidecar's Hy4 weight path (sidecar/peer_server_v4.py load_layer_experts + tensor_reader), with b12x
stubbed out (the b12x call itself only runs on a 6000: tests/b12x_layer_check_hy4.py)."""
import json
import os
import struct
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "models", "hy4", "config.json")

PROBE = r"""
import json, os, sys, types
root, model = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(root, "tests")); import conftest
for name in ("b12x", "b12x.moe", "b12x.moe.fused_moe", "b12x.preparation"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["b12x.preparation"].PreparationSession = object
sys.modules["b12x.preparation"].PreparedCall = object
sys.path.insert(0, os.path.join(root, "hook")); sys.path.insert(0, os.path.join(root, "sidecar"))
import torch, peer_server_v4 as ps, fp8_convert as fc
assert (ps.H, ps.I, ps.K, ps.E, ps.LIMIT) == (6144, 2048, 8, 256, 10.0), (ps.H, ps.I, ps.K, ps.E, ps.LIMIT)
t = ps.tensor_reader(model, raw=False)
w = t("model.layers.3.mlp.experts.5.gate_proj.weight"); s = t("model.layers.3.mlp.experts.5.gate_proj.weight_scale")
assert w.dtype == torch.float8_e4m3fn and tuple(w.shape) == (2048, 6144) and s.dtype == torch.float32 and tuple(s.shape) == (16, 48)
res = {}
for fmt in ("mxfp4", "mxfp8"):
    W = ps.load_layer_experts(t, 3, [5, 9], torch.device("cpu"), fmt)
    res[fmt] = {k: [list(v.shape), str(v.dtype)] for k, v in W.items()}
    # row 0 of expert 5's w13 is gate row 0, row 2048 is up row 0
    gate = fc.dequant_block_fp8(w, s)[0]
    d = fc.dequant(fmt, W["w13"][0], W["s13"][0])
    res[fmt]["gate_cos"] = float(torch.nn.functional.cosine_similarity(d[0].double(), gate.double(), dim=0))
print(json.dumps(res))
"""


def _write_safetensors(path, tensors):
    DT = {torch.float8_e4m3fn: "F8_E4M3", torch.float32: "F32"}
    header, blobs, off = {}, [], 0
    for name, t in tensors.items():
        b = t.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": DT[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_hy4_sidecar_loader(tmp_path):
    g = torch.Generator().manual_seed(0)
    T = {}
    for e in (5, 9):
        for p_, (n, k) in (("gate_proj", (2048, 6144)), ("up_proj", (2048, 6144)), ("down_proj", (6144, 2048))):
            T[f"model.layers.3.mlp.experts.{e}.{p_}.weight"] = (torch.randn(n, k, generator=g) * 50).clamp(
                -448, 448).to(torch.float8_e4m3fn)
            T[f"model.layers.3.mlp.experts.{e}.{p_}.weight_scale"] = torch.rand(n // 128, k // 128, generator=g) * 1e-3
    _write_safetensors(tmp_path / "model-00001-of-00001.safetensors", T)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in T}}))
    env = dict(os.environ, E3_CONFIG=CONFIG)
    out = subprocess.check_output([sys.executable, "-c", PROBE, ROOT, str(tmp_path)], env=env, text=True)
    r = json.loads(out.strip().splitlines()[-1])
    assert r["mxfp4"]["w13"] == [[2, 4096, 3072], "torch.uint8"] and r["mxfp4"]["s13"] == [[2, 4096, 192], "torch.uint8"]
    assert r["mxfp4"]["w2"] == [[2, 6144, 1024], "torch.uint8"] and r["mxfp4"]["s2"] == [[2, 6144, 64], "torch.uint8"]
    assert r["mxfp8"]["w13"] == [[2, 4096, 6144], "torch.uint8"] and r["mxfp8"]["w2"] == [[2, 6144, 2048], "torch.uint8"]
    assert r["mxfp4"]["gate_cos"] > 0.98 and r["mxfp8"]["gate_cos"] > 0.999
