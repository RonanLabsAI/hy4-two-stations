# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""CPU tests of hook/e3_hook.py on the Hy4 geometry, against a fake vLLM RoutedExperts module.

The fake reproduces the contract the hook relies on in vllm/vllm-openai:hy4 (thisjiang/vllm 5790aa8):
RoutedExperts.forward_modular(x, topk_weights, topk_ids, shared_experts, shared_experts_input) returns the routed sum
over the routes whose expert maps to a local slot in layer._expert_map (-1 = skipped, as the Triton FP8 path does with
ignore_invalid_experts + moe_sum(expert_map)). A fake sidecar computes the warm routes exactly, so serve mode must give
the stock (full-map) output bit for bit up to float addition order.
"""
import importlib
import os
import sys
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK = os.path.join(ROOT, "hook")
CONFIG = os.path.join(ROOT, "models", "hy4", "config.json")
TARGET = "vllm.model_executor.layers.fused_moe.routed_experts"


def _fake_vllm(monkeypatch, H, E):
    """Register a fake routed_experts module whose RoutedExperts honours _expert_map like the real modular path."""
    mod = types.ModuleType(TARGET)

    class RoutedExperts:
        def __init__(self, layer_name, ep_rank, ep_size, seed=0):
            g = torch.Generator().manual_seed(seed)
            n_local = E // ep_size
            self.layer_name = layer_name
            em = torch.full((E,), -1, dtype=torch.int32)
            em[ep_rank * n_local:(ep_rank + 1) * n_local] = torch.arange(n_local, dtype=torch.int32)
            self._expert_map = em
            self.w = torch.randn(n_local, H, H, generator=g) / H ** 0.5    # one linear map per local expert
            self.w13_weight = types.SimpleNamespace(_vllm_is_uva_offloaded=True, data_ptr=lambda: 0)
            self.quant_method = types.SimpleNamespace()
            self.calls = []

        def forward_modular(self, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
            self.calls.append(self._expert_map.clone())
            y = torch.zeros(x.shape[0], H)
            slot = torch.where(topk_ids >= 0, self._expert_map[topk_ids.clamp_min(0).long()], -1)
            for t in range(x.shape[0]):
                for k in range(topk_ids.shape[1]):
                    s = int(slot[t, k])
                    if s >= 0:
                        y[t] += topk_weights[t, k] * (x[t] @ self.w[s].T)
            return y

    mod.RoutedExperts = RoutedExperts
    monkeypatch.setitem(sys.modules, TARGET, mod)
    return mod


class FakePeer:
    """CPU stand-in for peer_tier_v4.PeerTierV4: send() records the packed warm rows, finish() adds the exact warm
    contribution computed from the worker's own expert weights (what a perfect sidecar returns)."""

    max_rows = 8192
    timeout = 1

    def __init__(self, layer_of, H):
        self.layer_of, self.H, self.sent = layer_of, H, []
        self.ok = torch.ones(1, dtype=torch.int32)

    def send(self, xs, wid, w, layer):
        self.sent.append((layer, wid.clone()))
        has = (wid >= 0).any(1)
        self._pending = (layer, xs.clone(), wid.clone(), w.clone())
        return has, torch.cumsum(has, 0) - 1, has.sum().reshape(1)

    def finish(self, ys, has, pos, rows):
        layer, xs, wid, w = self._pending
        lay = self.layer_of[layer]
        warm = lay._e3.warm
        for t in range(xs.shape[0]):
            for k in range(wid.shape[1]):
                if wid[t, k] >= 0:
                    g = warm[int(wid[t, k])]
                    s = int(lay._e3.map_full[g])
                    ys[t] += w[t, k] * (xs[t] @ lay.w[s].T)


def _load_hook(monkeypatch, tmp_path, mode, rowmap=None, count=""):
    monkeypatch.setenv("E3_CONFIG", CONFIG)
    monkeypatch.setenv("E3_MODE", mode)
    monkeypatch.setenv("E3_COUNT", count)
    monkeypatch.setenv("E3_CHECK", "0")
    if rowmap is not None:
        import json
        p = tmp_path / "rowmap.json"
        p.write_text(json.dumps(rowmap))
        monkeypatch.setenv("E3_ROWMAP", str(p))
    monkeypatch.syspath_prepend(HOOK)
    for m in ("e3_hook", "geometry", "peer_tier_v4"):
        sys.modules.pop(m, None)
    hook = importlib.import_module("e3_hook")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return hook


def test_layer_names(monkeypatch, tmp_path):
    hook = _load_hook(monkeypatch, tmp_path, "count")
    assert (hook.E, hook.TOPK, hook.HIDDEN) == (256, 8, 6144)
    assert hook._layer_idx("model.layers.5.mlp.experts") == 5
    assert hook._layer_idx("language_model.model.layers.77.mlp.experts") == 77
    assert hook._layer_idx("model.layers.5.mlp.experts.routed_experts") == 5
    assert hook._layer_idx("model.layers.0.mlp.experts") is None            # dense layer
    assert hook._layer_idx("model.layers.78.mlp.experts") is None
    assert hook._layer_idx("model.dfly_layers.2.mlp.experts") is None       # drafter
    assert hook._layer_idx("drafter.model.layers.3.mlp.experts") is None
    assert hook._layer_idx("mtp.layers.0.mlp.experts") is None
    assert hook._layer_idx("model.layers.5.self_attn") is None


def _setup_layers(mod, hook, ep_rank, layers, H):
    out = {}
    for i in layers:
        lay = mod.RoutedExperts(f"language_model.model.layers.{i}.mlp.experts", ep_rank, 2, seed=i)
        out[i] = lay
    return out


def _route(T, E, K, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)]).to(torch.int32)
    w = torch.rand(T, K, generator=g)
    return ids, w / w.sum(1, keepdim=True) * 2.827


def test_count_mode_needs_no_sidecar(monkeypatch, tmp_path):
    H, E, K = 32, 256, 8
    mod = _fake_vllm(monkeypatch, H, E)
    hook = _load_hook(monkeypatch, tmp_path, "count")
    hook.install()
    lay = mod.RoutedExperts("model.layers.3.mlp.experts", 1, 2)
    x = torch.randn(5, H)
    ids, w = _route(5, E, K, 0)
    y = mod.RoutedExperts.forward_modular(lay, x, w, ids)
    assert hook._peer is None and lay._e3.ready and lay._e3.warm == []
    assert torch.equal(lay._expert_map, lay.calls[-1])          # map untouched
    full = hook._orig["forward_modular"](lay, x, w, ids)
    assert torch.equal(y, full)


@pytest.mark.parametrize("ep_rank", [0, 1])
def test_serve_mode_masks_warm_and_adds_sidecar(monkeypatch, tmp_path, ep_rank):
    H, E, K = 32, 256, 8
    mod = _fake_vllm(monkeypatch, H, E)
    # warm ids on both ranks (global ids); each rank keeps only its own half
    rowmap = {"layers": {"3": {"warm": [1, 7, 100, 130, 200, 255]}, "4": {"warm": [0, 128]}}}
    hook = _load_hook(monkeypatch, tmp_path, "serve", rowmap)
    hook.install()
    lays = _setup_layers(mod, hook, ep_rank, (3, 4, 5), H)
    peer = FakePeer(lays, H)
    pt = types.SimpleNamespace(rank_warm=None, warm_hash=None, PATH="fake")
    real_pt = importlib.import_module("peer_tier_v4")
    pt.rank_warm, pt.warm_hash = real_pt.rank_warm, real_pt.warm_hash
    warm_all = pt.rank_warm(rowmap, ep_rank, E // 2)
    peer.serving = lambda: (True, pt.warm_hash(warm_all), ep_rank)
    peer.warm_count = lambda layer: len(warm_all.get(layer, []))
    monkeypatch.setattr(hook, "_init_peer", lambda device: peer)
    monkeypatch.setattr(hook, "_peer", peer)
    monkeypatch.setattr(hook, "_peer_mod", lambda: pt)
    for i, lay in lays.items():
        x = torch.randn(9, H)
        ids, w = _route(9, E, K, i)
        ids[0, :] = -1                                      # a padding row
        full = hook._orig["forward_modular"](lay, x, w, ids)
        y = mod.RoutedExperts.forward_modular(lay, x, w, ids)
        st = lay._e3
        assert st.warm == warm_all.get(i, [])
        if st.warm:
            # the GB300 call ran with the warm experts masked out of its map ...
            assert all(int(lay.calls[-1][g]) == -1 for g in st.warm)
            assert int((lay.calls[-1] >= 0).sum()) == E // 2 - len(st.warm)
            # ... the wire ids are the sorted positions of the warm ids, -1 elsewhere
            layer, wid = peer.sent[-1]
            assert layer == i
            for t in range(9):
                for k in range(K):
                    g = int(ids[t, k])
                    expect = st.warm.index(g) if g in st.warm else -1
                    assert int(wid[t, k]) == expect
        torch.testing.assert_close(y, full, rtol=1e-5, atol=1e-5)


def test_no_ep_identity_map(monkeypatch, tmp_path):
    """Single-Station layout (TP1, no expert map): with E3_ALLOW_NO_EP=1 the hook installs an identity map."""
    H, E, K = 16, 256, 8
    mod = _fake_vllm(monkeypatch, H, E)
    monkeypatch.setenv("E3_ALLOW_NO_EP", "1")
    hook = _load_hook(monkeypatch, tmp_path, "count", count=str(tmp_path / "counts.json"))
    monkeypatch.setattr(hook, "_start_count_dumper", lambda: None)
    hook.install()
    lay = mod.RoutedExperts("model.layers.2.mlp.experts", 0, 1)
    lay._expert_map = None
    x = torch.randn(3, H)
    ids, w = _route(3, E, K, 1)
    mod.RoutedExperts.forward_modular(lay, x, w, ids)
    assert torch.equal(lay._expert_map, torch.arange(E, dtype=torch.int32))
    # the routing counter saw all 3 x 8 routes, every expert local and (w13 offloaded) on the cold tier
    c = lay._e3.counts[2 if 3 > 64 else 1]
    assert int(c[hook.F_CNT:hook.F_CNT + E].sum()) == 3 * K
    assert int(c[hook.F_TR + 3]) == 3 * K and lay._e3.tier_info["cold"] == E
