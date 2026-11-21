#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""Memory budget for Hy4-FP8 on DGX Station GB300 (+ RTX PRO 6000), per EP rank, in GiB.

Layouts:
  B       EP2 over two Stations (TP2 + EP2): GB300 hot layers + Grace cold layers (UVA offload, layer granular)
  A       B plus a warm tier on each Station's 6000; the GB300 keeps every weight (warm experts are masked from the
          GB300's expert map, so the full-map fallback and the cosine gate keep working); A frees nothing on Grace
  single  one Station (TP1, EP1): GB300 hot + 6000 warm + Grace cold. Shown twice: with A's full-copy mechanism,
          and with an EXCLUSIVE warm tier (warm experts not stored on Grace at all; not implemented, see README)

Measured inputs (2026-11-20): GB300 256,703 MiB, RTX PRO 6000 Max-Q 97,887 MiB, Grace MemTotal 494.4 GiB with
~439 GiB MemAvailable before launch (both Stations). Checkpoint total 786,294,748,086 B (index metadata).
Estimates (flagged "est." in the table; the first boot's "Model loading took" and KV lines replace them):
non-expert weights per rank, KV budget, activation/graph/drafter workspace, 6000 reserve.

  hy4_budget.py [--config models/hy4/config.json] [--offload-gib 0=auto] [--warm-fmt mxfp4|mxfp8] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hook"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_rowmap as br  # noqa: E402
import geometry  # noqa: E402

GIB = 1 << 30
MIB = 1 << 20
GB300_GIB = 256703 * MIB / GIB          # nvidia-smi memory.total
RTX_GIB = 97887 * MIB / GIB
GRACE_AVAIL_GIB = 439.0                 # MemAvailable, idle Station with page cache dropped (2026-11-20: 439-443)
CKPT_BYTES = 786294748086


def offload_flag(k: int, n_local: int) -> float:
    """--cpu-offload-gb value that offloads exactly k whole MoE layers of n_local experts each.

    vLLM counts only the parameters it offloads (routed_experts.w13_weight + w2_weight, no scales) and checks
    `used >= budget` before each one, so the budget must be <= the bytes of k layers (it stops after layer k's w2) and
    > the bytes of k layers minus one w2 (or it would stop early). Flooring k layers' weight bytes to 0.01 GiB satisfies
    both whenever one w2 is larger than 0.01 GiB. Hy4 EP2: exactly 4.5 GiB per layer, so 37 layers = 166.5."""
    w13, w2, _, _ = br.expert_bytes()
    return int(k * n_local * (w13 + w2) / GIB * 100) / 100


def budget(geom, *, offload_gib: float, util: float = 0.95, kv_gib: float = 20.0, work_gib: float = 14.0,
           rtx_reserve_gib: float = 15.6, warm_fmt: str = "mxfp4", host_overhead_gib: float = 30.0) -> dict:
    eb_fp8 = geom.expert_bytes("fp8_block") / GIB
    eb_warm = geom.expert_bytes(warm_fmt) / GIB
    n_moe = len(geom.moe_layers)
    experts_total = n_moe * geom.n_experts * eb_fp8
    nonexpert_total = CKPT_BYTES / GIB - experts_total
    out = {"inputs": {"gb300_gib": round(GB300_GIB, 1), "rtx_gib": round(RTX_GIB, 1), "grace_avail_gib": GRACE_AVAIL_GIB,
                      "util": util, "kv_gib_est": kv_gib, "work_gib_est": work_gib, "rtx_reserve_gib_est": rtx_reserve_gib,
                      "host_overhead_gib_est": host_overhead_gib, "expert_fp8_mib": round(eb_fp8 * 1024, 2),
                      "expert_warm_mib": round(eb_warm * 1024, 2), "warm_fmt": warm_fmt,
                      "experts_total_gib": round(experts_total, 1), "nonexpert_total_gib": round(nonexpert_total, 1)}}

    # --- EP2 (B / A) ---------------------------------------------------------------------------------------------
    br.set_geometry(geom)
    n_local = geom.n_experts // 2
    per_layer = n_local * eb_fp8
    # TP2 shards attention, shared experts, dense MLP, embeddings and lm_head; the ViT, router gates, norms and iHC
    # parameters are counted as replicated (upper bound).
    nonexpert_rank = nonexpert_total / 2 + 2.5
    hbm_cap = GB300_GIB * util
    if offload_gib <= 0:   # auto: fewest whole Grace layers that let HBM fit
        k = max(0, n_moe - int((hbm_cap - nonexpert_rank - kv_gib - work_gib) // per_layer))
        offload_gib = offload_flag(k, n_local)
    grace = br.offloaded_layers(offload_gib, 2)
    hot_layers = n_moe - len(grace)
    hbm_used = nonexpert_rank + hot_layers * per_layer + kv_gib + work_gib
    out["inputs"]["offload_gib"] = offload_gib
    warm_cap = int((RTX_GIB - rtx_reserve_gib) / eb_warm)
    cold_experts = len(grace) * n_local
    warm = min(warm_cap, cold_experts - len(grace))       # leave >= 1 cold expert per Grace layer
    out["B"] = {"grace_layers": f"{grace[0]}-{grace[-1]}" if grace else "none", "n_grace_layers": len(grace),
                "hbm_layers": hot_layers, "hbm_nonexpert_est": round(nonexpert_rank, 1),
                "hbm_hot_experts": round(hot_layers * per_layer, 1), "hbm_kv_est": kv_gib, "hbm_work_est": work_gib,
                "hbm_total": round(hbm_used, 1), "hbm_cap": round(hbm_cap, 1), "hbm_fits": hbm_used <= hbm_cap,
                "grace_pinned": round(len(grace) * per_layer, 1),
                "grace_total": round(len(grace) * per_layer + host_overhead_gib, 1),
                "grace_fits": len(grace) * per_layer + host_overhead_gib <= GRACE_AVAIL_GIB,
                "cold_experts": cold_experts}
    out["A"] = dict(out["B"], warm_experts=warm, warm_cap=warm_cap,
                    rtx_warm=round(warm * eb_warm, 1), rtx_total=round(warm * eb_warm + rtx_reserve_gib, 1),
                    rtx_cap=round(RTX_GIB, 1), rtx_fits=warm * eb_warm + rtx_reserve_gib <= RTX_GIB,
                    warm_share_of_cold_experts=round(warm / max(cold_experts, 1), 3))

    # --- single Station (TP1, EP1) -------------------------------------------------------------------------------
    per_layer1 = geom.n_experts * eb_fp8
    nonexpert1 = nonexpert_total
    hot1 = int((hbm_cap - nonexpert1 - kv_gib - work_gib) // per_layer1)
    cold1_layers = n_moe - hot1
    warm1 = min(int((RTX_GIB - rtx_reserve_gib) / eb_warm), cold1_layers * (geom.n_experts - 1))
    grace_full = cold1_layers * per_layer1
    grace_excl = (cold1_layers * geom.n_experts - warm1) * eb_fp8
    hbm1 = nonexpert1 + hot1 * per_layer1 + kv_gib + work_gib
    out["single"] = {"offload_gib_flag": offload_flag(cold1_layers, geom.n_experts), "hbm_layers": hot1, "grace_layers": cold1_layers, "hbm_total": round(hbm1, 1),
                     "hbm_cap": round(hbm_cap, 1), "hbm_fits": hbm1 <= hbm_cap, "warm_experts": warm1,
                     "rtx_total": round(warm1 * eb_warm + rtx_reserve_gib, 1), "rtx_cap": round(RTX_GIB, 1),
                     "grace_full_copy": round(grace_full + host_overhead_gib, 1),
                     "grace_full_copy_fits": grace_full + host_overhead_gib <= GRACE_AVAIL_GIB,
                     "grace_exclusive": round(grace_excl + host_overhead_gib, 1),
                     "grace_exclusive_fits": grace_excl + host_overhead_gib <= GRACE_AVAIL_GIB,
                     "grace_exclusive_margin": round(GRACE_AVAIL_GIB - grace_excl - host_overhead_gib, 1)}
    return out


def table(b: dict) -> str:
    B, A, S, i = b["B"], b["A"], b["single"], b["inputs"]
    yn = lambda v: "fits" if v else "DOES NOT FIT"  # noqa: E731
    rows = [
        "| per EP rank (GiB) | B: EP2 GB300+Grace | A: B + 6000 warm | single: TP1 full-copy | single: TP1 exclusive warm |",
        "|---|---|---|---|---|",
        f"| --cpu-offload-gb | {i['offload_gib']} | same | {S['offload_gib_flag']} | n/a (needs exclusive loader) |",
        f"| MoE layers in HBM / Grace | {B['hbm_layers']} / {B['n_grace_layers']} ({B['grace_layers']}) | same | "
        f"{S['hbm_layers']} / {S['grace_layers']} | same |",
        f"| HBM: non-expert (est.) + hot experts + KV (est.) + work (est.) | {B['hbm_total']} of {B['hbm_cap']} "
        f"({yn(B['hbm_fits'])}) | same | {S['hbm_total']} of {S['hbm_cap']} ({yn(S['hbm_fits'])}) | same |",
        f"| 6000: warm experts ({i['warm_fmt']}) + reserve (est.) | - | {A['warm_experts']} experts, {A['rtx_total']} of "
        f"{A['rtx_cap']} ({yn(A['rtx_fits'])}) | {S['warm_experts']} experts, {S['rtx_total']} of {S['rtx_cap']} | same |",
        f"| Grace: pinned cold experts + host overhead (est.) | {B['grace_total']} of {i['grace_avail_gib']} "
        f"({yn(B['grace_fits'])}) | {A['grace_total']} (unchanged: full copy) | {S['grace_full_copy']} of "
        f"{i['grace_avail_gib']} ({yn(S['grace_full_copy_fits'])}) | {S['grace_exclusive']} of {i['grace_avail_gib']} "
        f"({yn(S['grace_exclusive_fits'])}, margin {S['grace_exclusive_margin']}) |",
        f"| warm share of Grace-resident experts | - | {A['warm_share_of_cold_experts']:.1%} | - | - |",
    ]
    notes = (f"\nExpert: {i['expert_fp8_mib']} MiB block-FP8 on the GB300 / Grace, {i['expert_warm_mib']} MiB "
             f"{i['warm_fmt']} on the 6000. Experts total {i['experts_total_gib']} GiB, non-expert "
             f"{i['nonexpert_total_gib']} GiB (checkpoint total minus experts). util {i['util']}, KV {i['kv_gib_est']}, "
             f"work {i['work_gib_est']}, 6000 reserve {i['rtx_reserve_gib_est']}, host overhead "
             f"{i['host_overhead_gib_est']} GiB are estimates.\n")
    return "\n".join(rows) + notes


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "models", "hy4", "config.json"))
    p.add_argument("--offload-gib", type=float, default=0.0, help="0 = auto (fewest Grace layers that fit HBM)")
    p.add_argument("--warm-fmt", default="mxfp4", choices=("mxfp4", "mxfp8"))
    p.add_argument("--kv-gib", type=float, default=20.0)
    p.add_argument("--json", default="")
    a = p.parse_args(argv)
    b = budget(geometry.load(a.config), offload_gib=a.offload_gib, warm_fmt=a.warm_fmt, kv_gib=a.kv_gib)
    print(table(b))
    if a.json:
        json.dump(b, open(a.json, "w"), indent=1)
    return b


if __name__ == "__main__":
    main()
