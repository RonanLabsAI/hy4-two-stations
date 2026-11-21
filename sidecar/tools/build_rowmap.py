#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
# 2026-11-20 (hy4-sidecar): geometry from --config (hook/geometry.py; default DeepSeek-V4-Pro), offload model over the
# MoE layers only (Hy4 layer 0 is dense), expert bytes in the checkpoint's format (Hy4: block FP8), warm-tier bytes in
# the sidecar's format (--warm-fmt), --capacity-gib as an alternative to --capacity.
"""Build the E3 three-tier placement (HBM hot / RTX PRO 6000 warm / Grace cold) per EP rank from routing counts.

Placement model (vLLM nightly-af7f9488, TP2+EP2, --cpu-offload-gb G, --cpu-offload-params routed_experts.w13_weight
routed_experts.w2_weight): the UVA offloader walks parameters in module order and offloads whole per-layer tensors
until the running total reaches G GiB (offloader/uva.py: the check is `bytes >= max` BEFORE each parameter). So the
first layers' local experts sit in Grace and the rest in HBM. On V4-Pro at G=200 that is layers 0-33 (200.81 GiB,
matching E1's "Total CPU offloaded parameters: 200.81 GiB").

Warm tier: per EP rank, the most-routed local experts of the Grace layers, ranked globally across layers by a
phase mix of normalised counts (default 0.6 decode + 0.4 prefill, as el8's mix-v1), up to the 6000's capacity.
HBM layers get no warm experts: moving an HBM expert to the 6000 frees nothing and adds a PCIe round trip.

  build_rowmap.py --counts calibration/routes-e1-gsm8k100.json --capacity 2400 --out rowmaps/rowmap-e3-gsm-v1.json
  build_rowmap.py --config models/hy4/config.json --counts e3-counts-rank0.json e3-counts-rank1.json \
      --offload-gib 165 --warm-fmt mxfp4 --capacity-gib 80 --out rowmaps/rowmap-hy4-v1.json
"""
import argparse
import json
import sys

import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hook"))
import geometry  # noqa: E402

GEOM = geometry.DSV4_PRO
E, LAYERS, HIDDEN, INTER = GEOM.n_experts, GEOM.n_layers, GEOM.hidden, GEOM.inter
FIRST = GEOM.first_dense
GIB = 1 << 30


def set_geometry(g) -> None:
    global GEOM, E, LAYERS, HIDDEN, INTER, FIRST
    GEOM = g
    E, LAYERS, HIDDEN, INTER, FIRST = g.n_experts, g.n_layers, g.hidden, g.inter, g.first_dense


def expert_bytes():
    """(w13, w2, s13, s2) bytes of one expert as the GB300 holds it (the checkpoint's format)."""
    if GEOM.weight_format == "fp8_block":
        bn, bk = GEOM.weight_block
        return (2 * INTER * HIDDEN, HIDDEN * INTER, 4 * 2 * (INTER // bn) * (HIDDEN // bk),
                4 * (HIDDEN // bn) * (INTER // bk))
    w13 = 2 * INTER * HIDDEN // 2          # FP4, two per byte
    w2 = HIDDEN * INTER // 2
    s13 = 2 * INTER * HIDDEN // 32         # E8M0 per 32
    s2 = HIDDEN * INTER // 32
    return w13, w2, s13, s2


def offload_trace(offload_gib: float, ep_size: int) -> list[tuple[int, str]]:
    """Every (layer, "w13"|"w2") parameter vLLM's UVA offloader would offload, in order (budget checked before each)."""
    w13, w2, _, _ = expert_bytes()
    n_local = E // ep_size
    budget, used, out = offload_gib * GIB, 0, []
    for layer in range(FIRST, LAYERS):
        for name, size in (("w13", w13 * n_local), ("w2", w2 * n_local)):
            if used >= budget:
                return out
            used += size
            out.append((layer, name))
    return out


def offloaded_layers(offload_gib: float, ep_size: int) -> list[int]:
    """Layers whose local w13 AND w2 are both UVA-offloaded (module order, budget checked before each param)."""
    w13, w2, _, _ = expert_bytes()
    n_local = E // ep_size
    budget, used, full = offload_gib * GIB, 0, []
    for layer in range(FIRST, LAYERS):
        got = 0
        for size in (w13 * n_local, w2 * n_local):
            if used >= budget:
                break
            used += size
            got += 1
        if got == 2:
            full.append(layer)
        if used >= budget:
            break
    return full


def load_counts(paths, mix):
    acc = np.zeros((LAYERS, E))
    for path in paths:
        d = json.load(open(path))
        phases = d.get("phases") or {"all": d["layers"]}
        for ph, layers in phases.items():
            w = mix.get(ph, mix.get("all", 1.0))
            arr = np.zeros((LAYERS, E))
            for k, v in layers.items():
                arr[int(k)] = np.asarray(v, float)
            tot = arr.sum(1, keepdims=True)
            tot[tot == 0] = 1
            acc += w * arr / tot
    return acc


def e2_score(acs_path, gsm_path):
    """R2.3 E2's score (eval/build_rowmap.py): 0.7 ACS-h1 decode + 0.3 GSM decode + 0.05 mean(prefill), each
    normalised per layer."""
    def arr(layers):
        a = np.zeros((LAYERS, E))
        for k, v in layers.items():
            a[int(k)] = np.asarray(v, float)
        return a

    def norm(a):
        t = a.sum(1, keepdims=True)
        t[t == 0] = 1
        return a / t
    acs = json.load(open(acs_path))["splits"]["acs-h1"]
    gsm = json.load(open(gsm_path))["phases"]
    return (0.7 * norm(arr(acs["decode"])) + 0.3 * norm(arr(gsm["decode"]))
            + 0.05 * (norm(arr(acs["prefill"])) + norm(arr(gsm["prefill"]))) / 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="", help="model config.json (default: DeepSeek-V4-Pro geometry)")
    p.add_argument("--warm-fmt", default="", help="sidecar weight format for warm_gb (default: checkpoint format)")
    p.add_argument("--capacity-gib", type=float, default=0.0,
                   help="warm experts per rank from a 6000 byte budget (overrides --capacity)")
    p.add_argument("--counts", nargs="*", default=[])
    p.add_argument("--mix", default="decode=0.6,prefill=0.4,all=1.0")
    p.add_argument("--offload-gib", type=float, default=200.0)
    p.add_argument("--ep-size", type=int, default=2)
    p.add_argument("--capacity", type=int, default=2400, help="warm experts per EP rank (6000 VRAM)")
    p.add_argument("--min-per-layer", type=int, default=8)
    p.add_argument("--layers", default="", help="override warm-eligible layers, e.g. 0-33")
    p.add_argument("--e2-rowmap", default="", help="E2 pin-hot rowmap: warm candidates = its COLD rows, all layers")
    p.add_argument("--acs", default="", help="routes-e1-acs.json (splits); with --gsm, use E2's score")
    p.add_argument("--gsm", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.config:
        set_geometry(geometry.load(a.config))
    warm_fmt = a.warm_fmt or GEOM.weight_format
    if a.capacity_gib:
        a.capacity = int(a.capacity_gib * GIB // GEOM.expert_bytes(warm_fmt))

    mix = {k: float(v) for k, v in (x.split("=") for x in a.mix.split(","))}
    if a.acs and a.gsm:
        score = e2_score(a.acs, a.gsm)
    else:
        score = load_counts(a.counts, mix)
    e2_hot = None
    if a.e2_rowmap:
        e2_hot = {int(k): set(v["hot"]) for k, v in json.load(open(a.e2_rowmap))["layers"].items()}
        score = score.copy()
        for layer, hot in e2_hot.items():
            score[layer, sorted(hot)] = -1.0      # never warm: already in HBM
    if e2_hot is not None:
        grace = sorted(e2_hot)
    elif a.layers:
        lo, hi = (int(x) for x in a.layers.split("-"))
        grace = list(range(lo, hi + 1))
    else:
        grace = offloaded_layers(a.offload_gib, a.ep_size)
    n_local = E // a.ep_size
    eb = GEOM.expert_bytes(warm_fmt)
    prior = not a.counts and not (a.acs and a.gsm)
    if prior:
        # No routing counts yet (a model's first rowmap, before any COUNT run): spread the capacity evenly over the
        # Grace layers instead of letting the zero-score tie-break fill the lowest layers first.
        a.min_per_layer = max(a.min_per_layer, min(a.capacity // max(len(grace), 1), n_local - 1))
        print(f"no counts: uniform prior, {a.min_per_layer} warm experts per Grace layer per rank", file=sys.stderr)
    warm = {layer: [] for layer in grace}
    report = {}
    for rank in range(a.ep_size):
        lo = rank * n_local
        sub = score[grace][:, lo:lo + n_local]
        chosen = np.zeros_like(sub, bool)
        # floor per layer first, so no Grace layer is left without a warm tier, then global greedy
        for i in range(len(grace)):
            top = [j for j in np.argsort(-sub[i], kind="stable") if sub[i, j] >= 0][: a.min_per_layer]
            chosen[i, top] = True
        rest = [(-sub[i, j], i, j) for i in range(len(grace)) for j in range(n_local) if not chosen[i, j]]
        rest.sort()
        rest = [r for r in rest if sub[r[1], r[2]] >= 0]
        # leave >= 1 cold (Grace) row per layer: a split layer with an empty cold set is untested in E2's hook
        cap = (sub >= 0).sum(1) - 1
        need = max(0, a.capacity - int(chosen.sum()))
        for _, i, j in rest:
            if need == 0:
                break
            if chosen[i].sum() < cap[i]:
                chosen[i, j] = True
                need -= 1
        for i, layer in enumerate(grace):
            warm[layer] += [lo + int(j) for j in np.nonzero(chosen[i])[0]]
        pos = np.clip(sub, 0, None)
        cov = float((pos * chosen).sum() / max(pos.sum(), 1e-12))
        per = chosen.sum(1)
        report[f"rank{rank}"] = {"warm_experts": int(chosen.sum()), "warm_gb": round(int(chosen.sum()) * eb / 1e9, 1),
                                 "insample_warm_share_of_grace_routes": round(cov, 4),
                                 "per_layer_min": int(per.min()), "per_layer_max": int(per.max())}
    hbm = [layer for layer in range(FIRST, LAYERS) if layer not in grace]
    out = {
        "format": "e3-rowmap/v1",
        "model": "deepseek-ai/DeepSeek-V4-Pro-0813" if GEOM.family == "deepseek_v4" else GEOM.model_type,
        "geometry": {"family": GEOM.family, "hidden": HIDDEN, "inter": INTER, "n_experts": E, "topk": GEOM.topk,
                     "n_layers": LAYERS, "first_dense": FIRST, "warm_fmt": warm_fmt},
        "e2_rowmap": a.e2_rowmap or None,
        "placement": {"ep_size": a.ep_size, "expert_placement": "linear", "offload_gib": a.offload_gib,
                      "grace_layers": grace, "hbm_layers": hbm,
                      "tiers": ("hot=E2 hot rows (HBM); warm=6000 (listed, from E2 cold rows); cold=E2 cold rows minus warm (Grace)" if a.e2_rowmap else "hot=HBM layers (all local experts); warm=6000 (listed); cold=Grace layers minus warm")},
        "source": {"counts": a.counts, "prior": "uniform" if prior else None, "mix": mix, "capacity_per_rank": a.capacity,
                   "min_per_layer": a.min_per_layer},
        "report": report,
        "layers": {str(layer): {"warm": sorted(v)} for layer, v in sorted(warm.items())},
    }
    json.dump(out, open(a.out, "w"), indent=1)
    json.dump({"grace_layers": f"{grace[0]}-{grace[-1]} ({len(grace)})", **report}, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
