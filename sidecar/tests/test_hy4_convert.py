# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""CPU tests for the Hy4 geometry and the block-FP8 -> MXFP4 / MXFP8 converter (python3 -m pytest tests/).

The real-weight test (test_real_layer_gate) runs when HY4_EXPERTS points at a safetensors file of Hy4-FP8 experts
(tools/fetch_experts.py writes one); otherwise it is skipped.
"""
import json
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hook"))
sys.path.insert(0, os.path.join(ROOT, "sidecar"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fp8_convert as fc  # noqa: E402
import geometry  # noqa: E402

HY4_CONFIG = os.path.join(ROOT, "models", "hy4", "config.json")


def test_hy4_geometry_from_config():
    g = geometry.load(HY4_CONFIG)
    assert (g.family, g.hidden, g.inter, g.n_experts, g.topk) == ("hy_v4", 6144, 2048, 256, 8)
    assert (g.n_layers, g.first_dense, g.n_shared) == (78, 1, 1)
    assert g.moe_layers == list(range(1, 78))
    assert g.swiglu_limit == 10.0 and abs(g.routed_scaling - 2.827) < 1e-9 and g.norm_topk_prob
    assert g.weight_format == "fp8_block" and tuple(g.weight_block) == (128, 128)
    assert g.extra["hc_mult"] == 4 and g.extra["dfly_num_hidden_layers"] == 5
    n = g.expert_names(5, 17)
    assert n["w1"] == "model.layers.5.mlp.experts.17.gate_proj.weight"
    assert n["w2s"] == "model.layers.5.mlp.experts.17.down_proj.weight_scale"
    # 77 MoE layers x 256 experts of FP8 = the ~743 GB of experts in the 786 GB checkpoint
    assert abs(77 * 256 * g.expert_bytes() / 1e9 - 744.3) < 0.5
    assert g.expert_bytes("mxfp4") == 3 * 2048 * 6144 // 2 + 3 * 2048 * 6144 // 32


def test_default_geometry_is_dsv4_pro(monkeypatch):
    monkeypatch.delenv("E3_CONFIG", raising=False)
    g = geometry.load()
    assert (g.hidden, g.inter, g.n_experts, g.topk, g.n_layers) == (7168, 3072, 384, 6, 61)


def test_geometry_vs_index_names():
    """Every expert tensor name the sidecar asks for exists in the checkpoint index (when it is available)."""
    idx = os.environ.get("HY4_INDEX", "")
    if not idx:
        pytest.skip("set HY4_INDEX=model.safetensors.index.json")
    wm = json.load(open(idx))["weight_map"]
    g = geometry.load(HY4_CONFIG)
    for layer in (g.first_dense, 40, g.n_layers - 1):
        for e in (0, g.n_experts - 1):
            assert all(n in wm for n in g.expert_names(layer, e).values())
    assert not any(n in wm for n in g.expert_names(0, 0).values())   # layer 0 is dense


def test_e2m1_roundtrip_and_nibble_order():
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6, 6] * 2, dtype=torch.float32)
    w = (grid * 2.0 ** -7).reshape(1, 32)
    packed, sc = fc.quant_mxfp4(w, "ceil")
    assert sc.item() == 127 - 7
    assert torch.equal(fc.dequant_mxfp4(packed, sc), w)
    # element 0 (0.0) in the low nibble of byte 0, element 1 (0.5 -> code 1) in the high nibble
    assert packed[0, 0].item() == (1 << 4) | 0
    assert packed[0, 1].item() == (3 << 4) | 2          # 1.5 -> code 3, 1.0 -> code 2


def test_e2m1_ties_to_even():
    a = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, 100.0])
    assert fc._e2m1_round(a).tolist() == [0, 2, 2, 4, 4, 6, 6, 7, 7]


def test_mxfp4_methods_and_error():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(256, 512, generator=g) * 0.02
    errs = {}
    for m in ("ocp", "ceil", "mse"):
        d = fc.dequant_mxfp4(*fc.quant_mxfp4(w, m))
        errs[m] = float(((d - w) ** 2).sum())
    assert errs["mse"] <= min(errs["ocp"], errs["ceil"]) + 1e-12
    rel = (errs["mse"] / float((w ** 2).sum())) ** 0.5
    assert 0.05 < rel < 0.15                               # E2M1 is coarse: ~10% relative error on Gaussian weights


def test_mxfp8_lossless_for_power_of_two_block_scales():
    g = torch.Generator().manual_seed(1)
    q = (torch.randn(256, 256, generator=g) * 60).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = torch.exp2(torch.randint(-12, -6, (2, 2), generator=g).float())
    w = fc.dequant_block_fp8(q.view(torch.uint8), s)
    codes, sc = fc.quant_mxfp8(w)
    d = fc.dequant_mxfp8(codes, sc)
    # power-of-two block scales: each 1x32 group is a power-of-two multiple of E4M3 codes, so only groups whose
    # rescale pushes small codes into E4M3's subnormal range can lose bits; nearly all elements are exact
    assert float((d == w).float().mean()) > 0.995
    assert float((d - w).norm() / w.norm()) < 2e-3


def test_block_dequant_shapes_and_ragged():
    q = torch.ones(300, 200, dtype=torch.float32).to(torch.float8_e4m3fn)
    s = torch.arange(1, 7, dtype=torch.float32).reshape(3, 2)
    d = fc.dequant_block_fp8(q, s)
    assert d.shape == (300, 200) and d[0, 0] == 1 and d[0, 199] == 2 and d[299, 0] == 5 and d[299, 199] == 6


def test_convert_expert_layout():
    g = torch.Generator().manual_seed(2)
    H, I = 256, 128

    def fp8(n, k):
        return (torch.randn(n, k, generator=g) * 50).clamp(-448, 448).to(torch.float8_e4m3fn), \
            torch.rand((n + 127) // 128, (k + 127) // 128, generator=g) * 1e-3 + 1e-4
    w1, s1 = fp8(I, H)
    w3, s3 = fp8(I, H)
    w2, s2 = fp8(H, I)
    for fmt, kdiv in (("mxfp4", 2), ("mxfp8", 1)):
        c = fc.convert_expert(w1, s1, w3, s3, w2, s2, fmt=fmt)
        assert c["w13"].shape == (2 * I, H // kdiv) and c["s13"].shape == (2 * I, H // 32)
        assert c["w2"].shape == (H, I // kdiv) and c["s2"].shape == (H, I // 32)
        assert all(t.dtype == torch.uint8 for t in c.values())
        # gate rows first ("w31"): row 0 of w13 dequantises close to gate row 0, not up row 0
        d = fc.dequant(fmt, c["w13"], c["s13"])
        gate0 = fc.dequant_block_fp8(w1, s1)[0]
        up0 = fc.dequant_block_fp8(w3, s3)[0]
        assert (d[0] - gate0).norm() < (d[0] - up0).norm()
        assert (d[I] - up0).norm() < (d[I] - gate0).norm()


def test_offline_gate_synthetic(tmp_path):
    """End-to-end offline gate on synthetic FP8 experts at a reduced geometry (the real one runs below)."""
    import offline_gate as og
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(3)
    H, I = 256, 128
    T = {}
    for e in range(4):
        for p, (n, k) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            T[f"model.layers.3.mlp.experts.{e}.{p}.weight"] = (torch.randn(n, k, generator=g) * 60).clamp(
                -448, 448).to(torch.float8_e4m3fn)
            T[f"model.layers.3.mlp.experts.{e}.{p}.weight_scale"] = torch.rand(n // 128, k // 128, generator=g) * 1e-3 + 1e-4
    f = tmp_path / "x.safetensors"
    save_file(T, str(f))
    cfg = json.load(open(HY4_CONFIG))
    cfg["text_config"].update(hidden_size=H, moe_intermediate_size=I)
    cfg.update(hidden_size=H, moe_intermediate_size=I)
    cp = tmp_path / "config.json"
    cp.write_text(json.dumps(cfg))
    out = tmp_path / "gate.json"
    rc4 = og.main(["--files", str(f), "--config", str(cp), "--fmt", "mxfp4", "--tokens", "16", "--json", str(out)])
    r4 = json.load(open(out))["results"][0]
    og.main(["--files", str(f), "--config", str(cp), "--fmt", "mxfp8", "--tokens", "16", "--json", str(out)])
    r8 = json.load(open(out))["results"][0]
    assert r4["layer"] == 3 and r4["top_k"] == 4
    assert r8["cos_mix_cand_vs_ref"] > r4["cos_mix_cand_vs_ref"] > 0.97
    assert rc4 in (0, 1)


@pytest.mark.skipif(not os.environ.get("HY4_EXPERTS"), reason="set HY4_EXPERTS=<safetensors of real Hy4 experts>")
def test_real_layer_gate():
    import offline_gate as og

    files = os.environ["HY4_EXPERTS"].split(",")
    geom = geometry.load(HY4_CONFIG)
    layers, norms = og.layer_tensors(files, geom)
    for layer, ex in sorted(layers.items()):
        r8 = og.gate_layer(layer, ex, geom, "mxfp8", 32, 0, norms.get(layer))
        r4 = og.gate_layer(layer, ex, geom, "mxfp4", 32, 0, norms.get(layer))
        assert r8["cos_mix_cand_vs_ref"] >= 0.996, r8
        assert r4["cos_mix_cand_vs_ref"] >= 0.98, r4
