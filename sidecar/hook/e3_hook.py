# Derived from original-el8/dgx-station-gb300-research @ 7699ae54, deepseek-v4.1-flash/m3/hook/mega_peer_hook.py
# Copyright 2026 Jason Cook. Licensed under the Apache License 2.0.
# Modifications Copyright 2026 RonanLabs (Apache License 2.0): rewritten for DeepSeek-V4-Pro-0813 in vLLM
# nightly-af7f9488, where deepseek_v4 without MegaMoE builds its experts with the generic FusedMoE factory
# (layers.N.ffn.experts.routed_experts, a RoutedExperts module) and the marlin MXFP4 kernel.
# Modifications 2026-11-20 (hy4-sidecar): geometry from hook/geometry.py (E3_CONFIG) instead of V4-Pro constants, Tencent
# Hy4 layer names (model.layers.N.mlp.experts), drafter (dfly / mtp) layers never hooked, count-only mode (E3_MODE=count:
# routing counts with no sidecar and no rowmap), the cosine reference runs without the shared experts.
"""E3 warm tier for DeepSeek-V4-Pro on TP2+EP2 GB300 Stations: RTX PRO 6000 sidecar per EP rank.

Three tiers per Station (per EP rank, local experts only):
  hot   HBM    - whatever the GB300 already holds in HBM (E1: positional, layers 34-60; E2: hot rows per layer)
  warm  6000   - the rowmap's "warm" experts of each hooked layer, computed by the sidecar on the RTX PRO 6000
  cold  Grace  - the rest of the UVA-offloaded experts, still read over C2C by the GB300's marlin kernel

How the warm tier is taken off the GB300 without touching weight placement: after loading, each hooked layer's
expert map (global id -> local slot, -1 = not mine) is copied and the warm experts are set to -1. vLLM's marlin path
(moe_align_block_size(ignore_invalid_experts=True) + ops.moe_sum(expert_map)) then skips those routes, so their
weights are never read from Grace. The warm routes go to the sidecar instead and come back as one add per token.
The GB300 keeps every weight, so:
  - E3_ENABLE=0, or a batch over E3_MAX_ROWS tokens, falls back to the full map = stock E1 numerics;
  - the cosine gate (E3_CHECK=N) compares the sidecar's rows against marlin on the same warm experts, using a
    warm-only map, on eager calls with T >= E3_CHECK_MIN_T. N is a per-layer budget, and a check with an all-zero
    reference (vLLM's dummy profile run) does not consume it.

Hy4 (vllm/vllm-openai:hy4 = thisjiang/vllm 5790aa8): the same class and method. HYV4MoEFused builds its experts with
FusedMoEFactory; the runner selects experts (sigmoid + bias, renormalised, routed_scaling_factor folded into
topk_weights: RoutedScaleMode.ROUTER), runs the shared experts itself, then calls RoutedExperts.forward_modular, whose
return value is the routed sum only. So the hook adds sum_k w_k * expert_k(x) with the router's own (scaled) weights,
and the shared expert, the TP / sequence-parallel reduction and the iHC post-mix all happen after it. The FP8 method
reads layer.expert_map on every apply (fused_moe_modular_method.py), so masking _expert_map works as on V4-Pro. It
needs a modular backend that honours expert_map and keeps UVA tensors in place: --moe-backend triton (TRTLLM is
monolithic and, like DeepGEMM, rewrites the weights into new GPU tensors at load).

Each EP rank's sidecar serves only that rank's local experts (linear placement: rank r owns [rE/2, (r+1)E/2)).
The rowmap lists global ids; each side keeps the ids it owns, sorted, and the sorted position is the wire id.
A sha256 of that per-layer list is the handshake: the worker refuses a sidecar built from a different rowmap.

Composes with E2: if an E2 hook splits a layer's local experts into an HBM module and a Grace module, apply this
hook to the layer's RoutedExperts module(s) the same way; the warm experts must then be absent from both.

Env:
  E3_HOOK=/path/e3_hook.py (sitecustomize installs it)   E3_ROWMAP=rowmap.json   E3_ENABLE=1
  E3_CHECK=0 (per-layer check budget)   E3_CHECK_MIN_T=256   E3_COUNT=/prof/e3-counts.json (route histogram)
  E3_COUNT also turns on the PC tier counters (2026-10-09): per layer and rank, bucketed by call size T (0: T == 1,
  1: 2 <= T <= 64, 2: T > 64), routes and unique experts per tier (remote / HBM hot / 6000 warm / Grace cold, from the
  LIVE maps the marlin calls use) and a histogram of unique Grace experts per call. Graph-safe (no host sync); host
  copies are refreshed on eager calls of the last hooked layer (E3_COUNT_COPY_SEC, default 1) and dumped every
  E3_COUNT_DUMP_SEC (default 3). Counters accumulate from boot, so read DELTAS between snapshots.
  E3_MAX_ROWS=8192   E3_SMALL_ROWS=64   E3_PEER_SPINS=20000000   E3_SHM=/dev/shm/e3_dsv4pro_peer
  E3_CONFIG=/model/config.json (geometry; unset = DeepSeek-V4-Pro)
  E3_MODE=serve (default: warm tier on the sidecar) | count (E3_COUNT only: no sidecar, no rowmap, stock numerics;
  the capture mode for a model's first rowmap)
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import re
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry as _geometry  # noqa: E402

GEOM = _geometry.load()
E = GEOM.n_experts                 # V4-Pro 384; Hy4 256
TOPK = GEOM.topk                   # V4-Pro 6; Hy4 8
HIDDEN = GEOM.hidden               # V4-Pro 7,168; Hy4 6,144
MODE = os.environ.get("E3_MODE", "serve").strip().lower()
if MODE not in ("serve", "count"):
    raise ValueError(f"E3_MODE={MODE!r}: expected serve or count")
ENABLE = os.environ.get("E3_ENABLE", "1") == "1"
ROWMAP_PATH = os.environ.get("E3_ROWMAP", "/e3/rowmap.json")
CHECK = int(os.environ.get("E3_CHECK", "0"))
CHECK_MIN_T = int(os.environ.get("E3_CHECK_MIN_T", "256"))
# vLLM's dummy profile run uses exactly max_num_batched_tokens (8192) tokens; keep it out of the gate.
CHECK_MAX_T = int(os.environ.get("E3_CHECK_MAX_T", "8191"))
COUNT_PATH = os.environ.get("E3_COUNT", "")
COUNT_COPY_SEC = float(os.environ.get("E3_COUNT_COPY_SEC", "1"))
COUNT_DUMP_SEC = float(os.environ.get("E3_COUNT_DUMP_SEC", "3"))
NB, HB = 3, 17                     # call-size buckets; unique-Grace-per-call histogram bins 0..15, 16+
F_CNT, F_TR, F_TU, F_H = 0, E, E + 4, E + 8
NF = E + 8 + HB                    # per bucket: counts[E] | routes per tier[4] | unique per tier[4] | hist[HB]
TIERS = ("remote", "hot", "warm", "cold")
STATS_PATH = os.environ.get("E3_STATS", "")
TARGET = "vllm.model_executor.layers.fused_moe.routed_experts"
# DS-V4: layers.N.ffn.experts[.routed_experts]; Hy4: [language_model.]model.layers.N.mlp.experts[.routed_experts]
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.(?:ffn|mlp)\.experts(?:\.routed_experts)?$")
_DRAFTER = re.compile(r"(?:^|\.)(?:mtp|dfly|dfly_layers|draft|drafter|eagle)(?:\.|_|$)")

_log = lambda m: sys.stderr.write(f"E3 {m}\n")  # noqa: E731

_rowmap = None
_peer = None
_pt = None
_orig = {}
_states: dict[int, "LayerState"] = {}
_count_host = None
_count_copied = 0.0
_check_log: list = []
_setup_warm_layers = 0  # layers whose expert map was masked: proves the A hook is active per layer


def _load_rowmap() -> dict:
    global _rowmap
    if _rowmap is None:
        with open(ROWMAP_PATH) as f:
            _rowmap = json.load(f)
        _log(f"rowmap {ROWMAP_PATH}: {len(_rowmap['layers'])} layers, "
             f"{sum(len(v.get('warm', [])) for v in _rowmap['layers'].values())} warm experts (both ranks)")
    return _rowmap


def _peer_mod():
    global _pt
    if _pt is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import peer_tier_v4

        _pt = peer_tier_v4
    return _pt


class LayerState:
    def __init__(self, idx: int):
        self.idx = idx
        self.ready = False
        self.warm: list[int] = []        # global ids, sorted, local to this rank
        self.lut = None                  # int32 [E]: global id -> wire id (sorted position) or -1
        self.map_full = None             # the layer's original expert map
        self.map_prod = None             # warm experts masked to -1
        self.map_warm = None             # only the warm experts (cosine reference)
        self.counts = None               # int64 [NB, NF] per-bucket expert counts + tier counters (E3_COUNT)
        self.tcode = None                # int64 [E] tier of each global id on this rank (0 remote 1 hot 2 warm 3 cold)
        self.iscold = None               # int64 [E] 1 where tcode == 3
        self.tier_info = None
        self.checks_left = CHECK
        self.offloaded = False
        self.e2 = None                   # R2.3 E2 PinState of this layer, when E2's split hook is installed
        self.e2_cold_full = self.e2_cold_prod = self.e2_cold_warm = self.e2_hot_none = None


def _layer_idx(name: str):
    """Decoder-layer index of a target-model RoutedExperts layer name; None for drafter (MTP / dfly) layers."""
    if _DRAFTER.search(name):
        return None
    m = _LAYER.search(name)
    if not m:
        return None
    idx = int(m.group(1))
    return idx if GEOM.first_dense <= idx < GEOM.n_layers else None


def _init_peer(device):
    global _peer
    if _peer is not None:
        return _peer
    pt = _peer_mod()
    peer = pt.PeerTierV4(device)
    for _ in range(600):  # the launcher starts the sidecar first; allow it up to 10 min to finish preparing
        ok, h, rank = peer.serving()
        if ok:
            break
        time.sleep(1)
    else:
        raise RuntimeError("E3: sidecar never reported serving on " + pt.PATH)
    _peer = peer
    _log(f"peer on {pt.PATH}: sidecar serving ep_rank={rank} hash={h:#x} max_rows={peer.max_rows} "
         f"timeout_spins={peer.timeout}")
    if STATS_PATH:
        _start_stats_dumper()
    return peer


def _setup(layer, st: LayerState, device) -> None:
    pt = _peer_mod()
    em = layer._expert_map
    if em is None:
        # No expert parallelism (single Station, TP1): give the layer an identity map so the warm experts can be
        # masked the same way. The modular Triton path takes an expert_map without EP (num_align_experts is the
        # global count either way); untested on GPU, so only on request.
        if os.environ.get("E3_ALLOW_NO_EP", "0") != "1":
            raise RuntimeError("E3 needs expert parallel (no expert map on " + layer.layer_name + "); "
                               "set E3_ALLOW_NO_EP=1 for the single-Station layout")
        em = torch.arange(E, dtype=torch.int32, device=device)
        layer._expert_map = em
    em_cpu = em.detach().to("cpu", torch.int64)
    local = (em_cpu >= 0).nonzero().flatten().tolist()
    n_local = len(local)
    ep_rank = local[0] // n_local
    if local != list(range(ep_rank * n_local, (ep_rank + 1) * n_local)):
        raise RuntimeError(f"E3: layer {st.idx} expert map is not linear placement")
    if n_local * (E // n_local) != E or em_cpu.numel() != E:
        raise RuntimeError(f"E3: layer {st.idx} expert map has {em_cpu.numel()} entries, geometry says {E} "
                           f"(E3_CONFIG={os.environ.get('E3_CONFIG', '')!r})")
    if MODE == "count":
        warm_all = {}
    else:
        rowmap = _load_rowmap()
        warm_all = pt.rank_warm(rowmap, ep_rank, n_local)
        peer = _init_peer(device)
        ok, h, srank = peer.serving()
        if srank != ep_rank or h != pt.warm_hash(warm_all):
            raise RuntimeError(f"E3: sidecar serves rank {srank} hash {h:#x}; this worker is rank {ep_rank} "
                               f"hash {pt.warm_hash(warm_all):#x} (rowmap {ROWMAP_PATH})")
    st.warm = warm_all.get(st.idx, [])
    if MODE != "count" and peer.warm_count(st.idx) != len(st.warm):
        raise RuntimeError(f"E3: layer {st.idx}: sidecar has {peer.warm_count(st.idx)} warm, worker {len(st.warm)}")
    lut = torch.full((E,), -1, dtype=torch.int32)
    for i, g in enumerate(st.warm):
        lut[g] = i
    st.lut = lut.to(device)
    st.map_full = em
    prod = em.clone()
    warm_only = torch.full_like(em, -1)
    if st.warm:
        w = torch.tensor(st.warm, dtype=torch.long, device=em.device)
        prod[w] = -1
        warm_only[w] = em[w]
    st.map_prod, st.map_warm = prod, warm_only
    e2 = sys.modules.get("dsv4pro_pin_hook")
    if e2 is not None and getattr(e2, "MODE", "") == "split":
        st2 = e2._STATES.get(layer.w13_weight.data_ptr())
        if st2 is None:
            raise RuntimeError(f"E3: E2 split hook is on but layer {st.idx} has no E2 state")
        st.e2 = st2
        st.e2_cold_full = st2.map_cold
        st.e2_hot_none = torch.full_like(st2.map_hot, -1)
        st.e2_cold_prod = st2.map_cold.clone()
        st.e2_cold_warm = torch.full_like(st2.map_cold, -1)
        if st.warm:
            w = torch.tensor(st.warm, dtype=torch.long, device=st2.map_cold.device)
            if bool((st2.map_hot[w] >= 0).any()) or bool((st2.map_cold[w] < 0).any()):
                raise RuntimeError(f"E3: layer {st.idx}: warm experts must be E2-cold (rowmap built without "
                                   "--e2-rowmap?)")
            st.e2_cold_prod[w] = -1
            st.e2_cold_warm[w] = st2.map_cold[w]
            st2.map_cold = st.e2_cold_prod
    if st.warm:
        layer._expert_map = st.map_prod
        global _setup_warm_layers
        _setup_warm_layers += 1
    st.offloaded = bool(getattr(getattr(layer, "w13_weight", None), "_vllm_is_uva_offloaded", False))
    if COUNT_PATH:
        st.counts = torch.zeros(NB, NF, dtype=torch.int64, device=device)
        _setup_tiers(st, em, device)
    st.ready = True
    _states[st.idx] = st
    if st.warm and not st.offloaded:
        _log(f"WARNING layer {st.idx}: {len(st.warm)} warm experts but w13 is NOT UVA-offloaded (HBM layer)")
    if st.idx in (GEOM.first_dense, GEOM.n_layers // 2, GEOM.n_layers - 1) or (st.warm and st.idx == min(warm_all)):
        _log(f"layer {st.idx} ({layer.layer_name}): ep_rank {ep_rank}, {n_local} local, {len(st.warm)} warm on "
             f"the 6000, w13 offloaded={st.offloaded}, quant={type(layer.quant_method).__name__}, "
             f"e2={'split' if st.e2 is not None else 'off'}")


_ONES = {}


def _ones(n, device):
    t = _ONES.get(device)
    if t is None or t.numel() < n:
        t = torch.ones(max(n, 8192 * TOPK), dtype=torch.int64, device=device)
        _ONES[device] = t
    return t[:n]


def _setup_tiers(st, em, device):
    """Tier of every global expert on this rank, from the maps the kernels will actually read (after E3 masking)."""
    local = em.detach().to("cpu", torch.int64) >= 0
    lut = st.lut.detach().to("cpu", torch.int64) >= 0
    code = torch.zeros(E, dtype=torch.int64)
    info = {}
    if st.e2 is not None:
        hot = st.e2.map_hot.detach().to("cpu", torch.int64) >= 0
        cold = st.e2.map_cold.detach().to("cpu", torch.int64) >= 0      # live (masked) cold map
        code[local & hot] = 1
        code[local & lut] = 2
        code[local & cold & ~lut & ~hot] = 3
        info = {"overlap_warm_cold": int((lut & cold).sum()), "overlap_hot_cold": int((hot & cold).sum()),
                "overlap_hot_warm": int((hot & lut).sum()), "local_unassigned": int((local & ~hot & ~lut & ~cold).sum()),
                "cold_w1_device": str(st.e2.cold["w1"].device), "e2_n_cold": int(st.e2.n_cold)}
    else:
        code[local] = 3 if st.offloaded else 1
        code[local & lut] = 2
    info.update({t: int((code == i).sum()) for i, t in enumerate(TIERS)})
    st.tier_info = info
    st.tcode = code.to(device)
    st.iscold = (code == 3).to(torch.int64).to(device)
    if info.get("overlap_warm_cold") or info.get("overlap_hot_cold") or info.get("local_unassigned"):
        _log(f"TIER WARNING layer {st.idx}: {info}")


def _count(st, topk_ids, T):
    """Graph-safe tier counting for one MoE call (T tokens)."""
    b = 0 if T == 1 else (1 if T <= 64 else 2)
    acc = st.counts[b]
    ids = topk_ids.flatten()
    idx = torch.where(ids >= 0, ids.long(), E)
    hitc = torch.zeros(E + 1, dtype=torch.int64, device=ids.device)
    hitc.index_add_(0, idx, _ones(idx.numel(), ids.device))
    h = hitc[:E]
    acc[F_CNT:F_CNT + E].add_(h)
    acc[F_TR:F_TR + 4].index_add_(0, st.tcode, h)
    u = (h > 0).to(torch.int64)
    acc[F_TU:F_TU + 4].index_add_(0, st.tcode, u)
    nc = (u * st.iscold).sum().clamp_max(HB - 1).view(1)
    acc[F_H:F_H + HB].index_add_(0, nc, _ones(1, ids.device))


def _start_count_dumper():
    """Write the pinned host copy of the counts every 20 s. The thread makes no CUDA calls (a CUDA call from another
    thread during vLLM's graph capture invalidates the capture)."""
    global _count_host
    _count_host = torch.zeros(GEOM.n_layers, NB, NF, dtype=torch.int64, device="cpu", pin_memory=True)

    def loop():
        while True:
            time.sleep(COUNT_DUMP_SEC)
            try:
                lay = [(i, s) for i, s in sorted(_states.items()) if s.counts is not None]
                hc = {i: _count_host[i].clone() for i, _ in lay}
                data = {str(i): hc[i][:, F_CNT:F_CNT + E].sum(0).tolist() for i, _ in lay}
                buckets = {str(b): {str(i): hc[i][b, F_CNT:F_CNT + E].tolist() for i, _ in lay} for b in range(NB)}
                tiers = {str(i): {str(b): {"routes": hc[i][b, F_TR:F_TR + 4].tolist(),
                                           "unique": hc[i][b, F_TU:F_TU + 4].tolist(),
                                           "hist_unique_cold": hc[i][b, F_H:F_H + HB].tolist()} for b in range(NB)}
                         for i, _ in lay}
                tmp = COUNT_PATH + ".tmp"
                with open(tmp, "w") as f:
                    json.dump({"format": "routing-counts/v1+pc-tiers", "source": "e3_hook E3_COUNT", "time": time.time(),
                               "mode": MODE, "geometry": {"family": GEOM.family, "n_experts": E, "topk": TOPK,
                                                          "n_layers": GEOM.n_layers},
                               "copied": _count_copied, "tier_names": list(TIERS),
                               "buckets": {"0": "T==1", "1": "2<=T<=64", "2": "T>64"},
                               "tier_info": {str(i): s.tier_info for i, s in lay},
                               "layers": data, "by_bucket": buckets, "tiers": tiers}, f)
                os.replace(tmp, COUNT_PATH)
            except Exception as exc:  # keep serving
                _log(f"count dump failed: {exc!r}")

    threading.Thread(target=loop, daemon=True, name="e3-count-dump").start()
    _log(f"route counting on, dumping to {COUNT_PATH}")


def _start_stats_dumper():
    def loop():
        while True:
            time.sleep(10)
            try:
                tmp = STATS_PATH + ".tmp"
                with open(tmp, "w") as f:
                    json.dump({"time": time.time(), "peer": _peer.counters(), "checks": _check_log[-512:],
                               "setup_warm_layers": _setup_warm_layers}, f)
                os.replace(tmp, STATS_PATH)
            except Exception as exc:
                _log(f"stats dump failed: {exc!r}")

    threading.Thread(target=loop, daemon=True, name="e3-stats-dump").start()


def _maybe_copy_counts(st) -> None:
    """Engine thread, eager calls of the last hooked layer only, at most every COUNT_COPY_SEC: async copy of every
    layer's counters to pinned host (stream-ordered, so the snapshot is consistent as of this call)."""
    global _count_copied
    if _count_host is None or torch.cuda.is_current_stream_capturing() or st.idx != max(_states):
        return
    now = time.time()
    if now - _count_copied < COUNT_COPY_SEC:
        return
    _count_copied = now
    for i, s in _states.items():
        if s.counts is not None:
            _count_host[i].copy_(s.counts, non_blocking=True)


def _call(layer, emap, x, topk_weights, topk_ids, shared_experts, shared_experts_input, e2_maps=None):
    """Run the stock path with a different expert map (and, under E2, different hot/cold maps). Eager only."""
    st = layer._e3
    saved = layer._expert_map
    layer._expert_map = emap
    e2_saved = None
    if st and st.e2 is not None and e2_maps is not None:
        e2_saved = (st.e2.map_hot, st.e2.map_cold)
        st.e2.map_hot, st.e2.map_cold = e2_maps
    try:
        return _orig["forward_modular"](layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
    finally:
        layer._expert_map = saved
        if e2_saved is not None:
            st.e2.map_hot, st.e2.map_cold = e2_saved


def _forward_modular(self, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
    st = getattr(self, "_e3", None)
    if st is None:
        idx = _layer_idx(getattr(self, "layer_name", ""))
        st = LayerState(idx) if idx is not None else False
        self._e3 = st
    if st is False or not ENABLE:
        return _orig["forward_modular"](self, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
    capturing = torch.cuda.is_current_stream_capturing()
    if not st.ready:
        if capturing:
            raise RuntimeError(f"E3: layer {st.idx} first seen during CUDA graph capture; cannot set up there")
        if COUNT_PATH or (MODE == "serve" and st.idx in {int(k) for k in _load_rowmap()["layers"]}):
            _setup(self, st, x.device)
            if COUNT_PATH and _count_host is None:
                _start_count_dumper()
        else:
            st.ready = True
    T = x.shape[0]
    if st.counts is not None and T > 0:
        _count(st, topk_ids, T)
        _maybe_copy_counts(st)
    if not st.warm or T == 0 or MODE == "count":
        return _orig["forward_modular"](self, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
    if self._expert_map is not st.map_prod:
        # Something re-registered the expert map (EPLB, a config reset): re-mask before it double-counts.
        if capturing:
            raise RuntimeError(f"E3: layer {st.idx} expert map replaced; refusing to capture")
        _log(f"layer {st.idx}: expert map was replaced; re-masking warm experts")
        st.map_full = self._expert_map
        st.map_prod = st.map_full.clone()
        st.map_prod[torch.tensor(st.warm, device=st.map_full.device)] = -1
        self._expert_map = st.map_prod
    peer = _peer
    if T > peer.max_rows:
        # Too many rows for the shared buffer: the GB300 still holds the warm weights, so run stock E1 numerics.
        e2m = (st.e2.map_hot, st.e2_cold_full) if st.e2 is not None else None
        return _call(self, st.map_full, x, topk_weights, topk_ids, shared_experts, shared_experts_input, e2m)

    xs = x[:, :HIDDEN] if x.shape[-1] != HIDDEN else x
    ids = topk_ids.clamp_min(0).long()
    wid = torch.where(topk_ids >= 0, st.lut[ids], -1)
    has, pos, rows = peer.send(xs, wid, topk_weights, st.idx)
    y = _orig["forward_modular"](self, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
    if not isinstance(y, torch.Tensor):
        raise RuntimeError(f"E3: layer {st.idx}: routed experts returned {type(y)} (deferred finalize?); "
                           "the add needs a finalized tensor")
    check = st.checks_left > 0 and CHECK_MIN_T <= T <= CHECK_MAX_T and not capturing
    ys = y[:, :HIDDEN] if y.shape[-1] != HIDDEN else y
    peer.finish(ys, has, pos, rows)
    if check:
        _cosine_check(self, st, x, topk_weights, topk_ids, shared_experts, shared_experts_input, has, pos, rows, y)
    return y


def _cosine_check(layer, st, x, topk_weights, topk_ids, se, sei, has, pos, rows, y):
    peer = _peer
    torch.cuda.synchronize()
    T = x.shape[0]
    e2m = (st.e2_hot_none, st.e2_cold_warm) if st.e2 is not None else None
    # No shared experts in the reference: it must be the warm routes only, and a second run of the runner's
    # SharedExperts object would disturb its overlap bookkeeping.
    ref = _call(layer, st.map_warm, x, topk_weights, topk_ids, None, None, e2m)
    ref = (ref[:, :HIDDEN] if ref.shape[-1] != HIDDEN else ref).float()
    ref_norm = float(ref.norm())
    n_rows = int(rows.item())
    if ref_norm < 1e-6 or n_rows == 0:
        return  # dummy profile input or no warm route: does not consume the budget
    ok = int(peer.ok.item())
    side = peer.dense_rows(T, has, pos) if ok else torch.zeros_like(ref)
    diff = (side - ref).flatten()
    cos = float(torch.nn.functional.cosine_similarity(side.flatten(), ref.flatten(), dim=0))
    rel = float(diff.norm() / max(ref_norm, 1e-9))
    share = ref_norm / max(float(y.float().norm()), 1e-9)
    # Identity check (review 2026-11-21): the warm-only cosine above cannot see a double-count (masking ignored) or
    # a partial mask. So also run the production (masked) map and the full map, both without shared experts, and
    # require cos(prod + sidecar, full) to pass and the masked output to differ from the full one.
    e2f = (st.e2.map_hot, st.e2_cold_full) if st.e2 is not None else None
    prod = _call(layer, st.map_prod, x, topk_weights, topk_ids, None, None, None)
    full = _call(layer, st.map_full, x, topk_weights, topk_ids, None, None, e2f)
    prod = (prod[:, :HIDDEN] if prod.shape[-1] != HIDDEN else prod).float()
    full = (full[:, :HIDDEN] if full.shape[-1] != HIDDEN else full).float()
    id_cos = float(torch.nn.functional.cosine_similarity((prod + side).flatten(), full.flatten(), dim=0))
    prod_frac = float(prod.norm()) / max(float(full.norm()), 1e-9)
    st.checks_left -= 1
    rec = {"layer": st.idx, "T": T, "rows": n_rows, "ok": ok, "cos": round(cos, 6), "rel": round(rel, 5),
           "warm_share": round(share, 4), "id_cos": round(id_cos, 6), "prod_frac": round(prod_frac, 5)}
    _check_log.append(rec)
    _log(f"check layer={st.idx} T={T} rows={n_rows} ok={ok} cos={cos:.6f} rel_diff={rel:.5f} "
         f"(warm share of routed output norm {share:.3f}) id_cos={id_cos:.6f} prod_frac={prod_frac:.5f}")


def _patch(module):
    cls = module.RoutedExperts
    if getattr(cls, "_e3_patched", False):
        return
    _orig["forward_modular"] = cls.forward_modular
    cls.forward_modular = _forward_modular
    cls._e3_patched = True
    _log(f"patched {TARGET}.RoutedExperts.forward_modular (mode={MODE}, enable={ENABLE}, check={CHECK}/layer, "
         f"min_t={CHECK_MIN_T}, count={'on' if COUNT_PATH else 'off'}, geometry {GEOM.family} E={E} top-k={TOPK} "
         f"H={HIDDEN} layers={GEOM.n_layers})")


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, target, patch):
        self.target, self.patch = target, patch

    def find_spec(self, name, path, target=None):
        if name != self.target:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig_exec=orig_exec, _patch_fn=self.patch):
            _orig_exec(module)
            _patch_fn(module)

        spec.loader.exec_module = exec_module
        return spec


def install():
    if TARGET in sys.modules:
        _patch(sys.modules[TARGET])
    else:
        sys.meta_path.insert(0, _Finder(TARGET, _patch))
