# ruff: noqa: E741
#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""Offline critical-path simulator for NOVEL-DIRECTIONS-CORTEX.md N1 and N2 on DS-V4-Pro E3-A16 (TP2+EP2).

N1  replicated cold tier: each Grace also holds a hit-weighted replica set of the OTHER rank's cold experts;
    per layer, both ranks run the same deterministic assignment of replicated cold picks to the less-loaded rank.
    TP-replicated activations + EP partials summed in the existing TP all-reduce => no new communication.
N1s (sim extension, not in the NOVEL doc): a replicated expert may also be SPLIT by intermediate columns, half
    computed on each rank (partials still sum in the same all-reduce).
N2  stall-only rerouting: a cold pick whose selection score is within eps of the best non-selected RESIDENT
    expert (HBM hot or 6000 warm, either rank) is swapped, but only when that lowers the layer's cross-rank max.

Step model (per decode step, C1 = one token):
    step = F + sum_l max_r t_grace(n[l, r]) + overheads
    t_grace(n) = EMPTY if n == 0 else A + B * n       (n = experts this rank reads from Grace in layer l)
F is everything rank-symmetric (dense GEMM, mHC, norm/quant, attention, routing, hot marlin, sidecar, collective
transfer, idle) and is fitted so the baseline reproduces the R3.1 nsys step (22.59 ms). A, B, EMPTY are fitted to
the per-call cold marlin durations in the R3.1 / R3.3 nsys traces (see fit_cost()).

DATA PROVENANCE. No per-token routes or gate scores survive (R3.3 deleted them). The route generator is
SYNTHETIC: per layer, top-6 sampled without replacement in proportion to the measured per-expert decode counts
(Gumbel-top-k), with cold experts' odds tilted by KAPPA so the per-rank Grace hit rate matches the R3.1 trace.
Gate scores are a random-utility model (u = 1 + RHO * (log p + Gumbel - m)); RHO (router noise relative to the
selected score level) is NOT identifiable from counts, so N2 is reported as a function of RHO.
Inputs are aggregate per-expert counts only (no text, no prompt rows); keep outputs out of public repos anyway.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import numpy as np

E, LAYERS, TOPK, NLOC = 384, 61, 6, 192
HASH_LAYERS = (0, 1, 2)                    # routed by token-id hash: no gate score to trade, never rerouted
EXPERT_GRACE_BYTES = 33.03e6               # marlin MXFP4 w13+w2 weights (R3.3 G2); scales live in HBM
EXPERT_SCALE_BYTES = 2 * 3072 * 7168 // 32 + 7168 * 3072 // 32   # E8M0 per 32 -> 2.06 MB/expert, HBM
GIB = 1 << 30
ROOT = pathlib.Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------------------


def _arr(layers: dict) -> np.ndarray:
    a = np.zeros((LAYERS, E))
    for k, v in layers.items():
        a[int(k)] = np.asarray(v, float)
    return a


def _norm(a: np.ndarray) -> np.ndarray:
    t = a.sum(1, keepdims=True)
    t[t == 0] = 1
    return a / t


def load_counts(root: pathlib.Path = ROOT) -> dict[str, np.ndarray]:
    acs = json.load(open(root / "calibration/routes-e1-acs.json"))["splits"]
    gsm = json.load(open(root / "calibration/routes-e1-gsm8k100.json"))["phases"]
    return {"acs_h1": _norm(_arr(acs["acs-h1"]["decode"])), "acs_h2": _norm(_arr(acs["acs-h2"]["decode"])),
            "gsm": _norm(_arr(gsm["decode"]))}


def load_placement(root: pathlib.Path = ROOT, warm_on: bool = True) -> np.ndarray:
    """tier[l, e]: 0 = HBM hot, 1 = 6000 warm, 2 = Grace cold. Owner rank of expert e = e // 192 (linear EP)."""
    hot = json.load(open(root / "calibration/rowmap-dsv4pro-acs-v1.json"))["layers"]
    warm = json.load(open(root / "rowmaps/rowmap-e3-on-e2-acs-v1.json"))["layers"]
    tier = np.full((LAYERS, E), 2, np.int8)
    for l in range(LAYERS):
        tier[l, hot[str(l)]["hot"]] = 0
        if warm_on:
            tier[l, warm[str(l)]["warm"]] = 1
    return tier


def fit_cost(paths: list[pathlib.Path]) -> dict:
    """Fit t = A + B*n on the per-call cold marlin durations (n = experts, recovered by clustering)."""
    vals = []
    for p in paths:
        d = json.load(open(p))
        vals += [x for r in ("0", "1") for s in d[r]["cold"] for x in s]
    v = np.asarray(vals)
    empty = float(np.median(v[v < 0.01]))
    hit = v[v >= 0.01]
    # first pass: n from a rough 0.09 ms/expert step, then least squares
    n = np.clip(np.round((hit - 0.045) / 0.088), 1, None)
    B, A = np.polyfit(n, hit, 1)
    n = np.clip(np.round((hit - A) / B), 1, None)
    B, A = np.polyfit(n, hit, 1)
    resid = hit - (A + B * n)
    return {"A": float(A), "B": float(B), "EMPTY": empty, "n_calls": int(len(hit)),
            "resid_p90_us": float(np.quantile(np.abs(resid), 0.9) * 1e3),
            "eff_gbps": EXPERT_GRACE_BYTES / (B * 1e-3) / 1e9}


def trace_stats(path: pathlib.Path, cost: dict) -> dict:
    d = json.load(open(path))
    c0, c1 = np.asarray(d["0"]["cold"]), np.asarray(d["1"]["cold"])
    thr = 0.01
    n0 = np.where(c0 > thr, np.clip(np.round((c0 - cost["A"]) / cost["B"]), 1, None), 0)
    n1 = np.where(c1 > thr, np.clip(np.round((c1 - cost["A"]) / cost["B"]), 1, None), 0)
    crit = np.maximum(c0, c1).sum(1)
    return {"steps": int(len(c0)), "crit_mean": float(crit.mean()), "crit_median": float(np.median(crit)),
            "per_rank_mean": [float(c0.sum(1).mean()), float(c1.sum(1).mean())],
            "hit_layers_rank": [float((c0 > thr).sum(1).mean()), float((c1 > thr).sum(1).mean())],
            "hit_layers_any": float(((c0 > thr) | (c1 > thr)).sum(1).mean()),
            "hit_layers_both": float(((c0 > thr) & (c1 > thr)).sum(1).mean()),
            "cold_experts_rank_step": [float(n0.sum(1).mean()), float(n1.sum(1).mean())],
            "n": np.stack([n0, n1], -1)}   # [step, layer, rank] experts read from Grace (measured)


# ---------------------------------------------------------------------------------------------------------------
# synthetic routes
# ---------------------------------------------------------------------------------------------------------------


def gen_routes(p: np.ndarray, tier: np.ndarray, kappa: float, n_tok: int, rng: np.random.Generator):
    """Return ids [n_tok, L, 6], g [n_tok, L, E] (Gumbel utilities, for N2 margins).
    Cold experts' odds are multiplied by kappa (calibration tilt)."""
    logp = np.log(np.maximum(p, 1e-9)) + np.where(tier == 2, np.log(kappa), 0.0)
    g = (logp[None] + rng.gumbel(size=(n_tok, LAYERS, E))).astype(np.float32)
    ids = np.argpartition(-g, TOPK, axis=2)[:, :, :TOPK]
    return ids, g


# ---------------------------------------------------------------------------------------------------------------
# per layer-step decision
# ---------------------------------------------------------------------------------------------------------------


class Cost:
    def __init__(self, A, B, EMPTY, half_eff=1.0):
        self.A, self.B, self.EMPTY, self.half_eff = A, B, EMPTY, half_eff

    def t(self, n: float) -> float:
        return self.EMPTY if n <= 0 else self.A + self.B * n


def layer_time(cost: Cost, picks, removed=frozenset(), mode="base"):
    """picks: list of (home_rank, replicated). Returns (crit_ms, n0, n1) for the best rank-identical assignment.
    mode base: everything at home. n1: replicated picks may move. n1s: replicated picks may move or split.
    n1h: replicated picks stay home or split (the replica only needs the half the other rank computes)."""
    fixed = [0.0, 0.0]
    mov = []
    for i, (h, rep) in enumerate(picks):
        if i in removed:
            continue
        if mode != "base" and rep:
            mov.append(h)
        else:
            fixed[h] += 1
    if not mov:
        return max(cost.t(fixed[0]), cost.t(fixed[1])), fixed[0], fixed[1]
    choices = {"n1s": (0, 1, 2), "n1h": ("home", 2)}.get(mode, (0, 1))
    best = None
    half = 0.5 / cost.half_eff
    if len(mov) > 6:                       # never at C1; keep it bounded (greedy LPT)
        n = list(fixed)
        for _ in mov:
            n[0 if n[0] <= n[1] else 1] += 1
        return max(cost.t(n[0]), cost.t(n[1])), n[0], n[1]
    for combo in itertools.product(choices, repeat=len(mov)):
        n = list(fixed)
        combo = tuple(h if c == "home" else c for c, h in zip(combo, mov))
        for c in combo:
            if c == 2:
                n[0] += half
                n[1] += half
            else:
                n[c] += 1
        tt = max(cost.t(n[0]), cost.t(n[1]))
        key = (round(tt, 9), sum(1 for c in combo if c == 2))
        if best is None or key < best[0]:
            best = (key, tt, n[0], n[1])
    return best[1], best[2], best[3]


# ---------------------------------------------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------------------------------------------


def replica_sets(sel_p: np.ndarray, tier: np.ndarray, budget_gib: float, kappa: float = 1.0,
                 frac: float = 1.0) -> tuple[np.ndarray, dict]:
    """Per host, replicate the other rank's cold experts with the highest selection-set hit mass, up to budget.
    Selection uses acs_h1 + gsm (the rowmap's own inputs), evaluation uses acs_h2 + gsm: replica choice is out of
    sample for ACS. Returns rep[l, e] (True = a replica exists on the non-home rank) and a capacity report."""
    rep = np.zeros((LAYERS, E), bool)
    per_exp = frac * EXPERT_GRACE_BYTES / GIB
    n_max = int(budget_gib / per_exp + 1e-9)
    rep_ct = []
    for home in (0, 1):
        sl = slice(home * NLOC, (home + 1) * NLOC)
        cand = [(-sel_p[l, home * NLOC + j], l, home * NLOC + j)
                for l in range(LAYERS) for j in range(NLOC) if tier[l, home * NLOC + j] == 2]
        cand.sort()
        for _, l, e in cand[:n_max]:
            rep[l, e] = True
        rep_ct.append(min(n_max, len(cand)))
        del sl
    n_cold = int((tier == 2).sum() / 2)
    return rep, {"replicas_per_host": rep_ct, "cold_per_rank": n_cold,
                 "grace_gib_per_host": [c * per_exp for c in rep_ct],
                 "hbm_scale_gib_per_rank_if_scales_in_hbm": [c * frac * EXPERT_SCALE_BYTES / GIB for c in rep_ct]}


def simulate(ids, g, tier, rep, cost: Cost, mode="base", eps=None, rho=0.1, budget=None,
             tok_per_step=1) -> dict:
    """ids [S*T, L, 6] (T tokens per step), g utilities. Returns per-step crit and N2 quality proxies."""
    n_tok = ids.shape[0]
    S = n_tok // tok_per_step
    crit = np.zeros(S)
    rerouted_picks = 0
    steps_with_reroute = 0
    mass_changed = []          # per token-layer sum of normalized weights of rerouted picks (only where >0)
    hit_any_layers = 0
    stall_layers_removed = 0
    for s in range(S):
        toks = range(s * tok_per_step, (s + 1) * tok_per_step)
        layer_opts = []        # (gain, n_subs, layer, subset, picks_meta)
        base_t = np.zeros(LAYERS)
        new_t = np.zeros(LAYERS)
        for l in range(LAYERS):
            # union of cold experts across this step's tokens; per distinct expert keep the min margin over tokens
            meta = {}
            for t in toks:
                for e in ids[t, l]:
                    e = int(e)
                    if tier[l, e] != 2:
                        continue
                    m = meta.setdefault(e, {"tok": [], "margin": np.inf})
                    m["tok"].append(t)
            if not meta:
                base_t[l] = new_t[l] = cost.EMPTY
                continue
            hit_any_layers += 1
            elist = sorted(meta)
            picks = [(e // NLOC, bool(rep[l, e])) for e in elist]
            t0, _, _ = layer_time(cost, picks, mode="base")
            base_t[l] = t0
            mode_n1 = {"n1": "n1", "n1s": "n1s", "n1h": "n1h", "n1+n2": "n1", "n1s+n2": "n1s"}.get(mode, "base")
            tb, _, _ = layer_time(cost, picks, mode=mode_n1)
            new_t[l] = tb
            if eps is None or l in HASH_LAYERS:
                continue
            # N2: an expert is reroutable only if, for EVERY token that picked it, it is within eps of that
            # token's best non-selected resident alternative (else that token still needs it from Grace)
            elig = []
            for i, e in enumerate(elist):
                ok = True
                wsum = 0.0
                for t in meta[e]["tok"]:
                    gt = g[t, l]
                    sel = ids[t, l]
                    m_lvl = np.sort(gt[sel])[0]
                    u = 1.0 + rho * (gt - m_lvl)
                    mask = np.ones(E, bool)
                    mask[sel] = False
                    mask &= tier[l] != 2
                    u_alt = u[mask].max()
                    rel = (u[e] - u_alt) / max(u[e], 1e-6)
                    if rel > eps:
                        ok = False
                        break
                    us = np.maximum(u[sel], 0.01)
                    wsum += u[e] / us.sum() if u[e] > 0 else 0.0
                if ok:
                    elig.append((i, wsum))
            if not elig:
                continue
            best = (tb, 0, frozenset(), 0.0)
            for k in range(1, len(elig) + 1):
                for sub in itertools.combinations(elig, k):
                    rm = frozenset(i for i, _ in sub)
                    tt, _, _ = layer_time(cost, picks, rm, mode=mode_n1)
                    w = sum(x for _, x in sub)
                    if (round(tt, 9), k, w) < (round(best[0], 9), best[1] if best[1] else 99, best[3]):
                        best = (tt, k, rm, w)
            if best[1] and best[0] < tb - 1e-9:
                layer_opts.append((tb - best[0], best[1], l, best[0], best[3]))
        # apply N2 options under the per-step substitution budget (greedy by gain per substitution)
        used = 0
        any_rr = False
        for gain, k, l, tt, w in sorted(layer_opts, key=lambda x: -x[0] / x[1]):
            if budget is not None and used + k > budget:
                continue
            used += k
            new_t[l] = tt
            rerouted_picks += k
            mass_changed.append(w)
            stall_layers_removed += 1 if tt <= cost.EMPTY + 1e-9 else 0
            any_rr = True
        steps_with_reroute += any_rr
        crit[s] = new_t.sum()
    n_tok_layers = S * tok_per_step * (LAYERS - len(HASH_LAYERS))
    return {"crit_mean": float(crit.mean()), "crit_p50": float(np.median(crit)), "steps": S,
            "hit_layers_any": hit_any_layers / S,
            "reroute_steps_frac": steps_with_reroute / S,
            "rerouted_picks_frac": rerouted_picks / (S * tok_per_step * LAYERS * TOPK),
            "rerouted_per_step": rerouted_picks / S,
            "mass_changed_per_token_layer": (sum(mass_changed) / n_tok_layers) if mass_changed else 0.0,
            "mass_changed_per_rerouted_pick": (sum(mass_changed) / rerouted_picks) if rerouted_picks else 0.0,
            "stall_layers_removed_per_step": stall_layers_removed / S}


def base_stats(ids, tier, cost: Cost) -> dict:
    """Vectorized baseline (C1): per-rank hit layers, cold experts, per-rank and crit Grace ms."""
    cold = tier[np.arange(LAYERS)[None, :, None], ids] == 2          # [T, L, 6]
    rank = ids // NLOC
    n0 = (cold & (rank == 0)).sum(2)
    n1 = (cold & (rank == 1)).sum(2)
    tf = np.vectorize(cost.t)
    t0, t1 = tf(n0), tf(n1)
    return {"hit_layers_rank": [float((n0 > 0).sum(1).mean()), float((n1 > 0).sum(1).mean())],
            "hit_layers_any": float(((n0 > 0) | (n1 > 0)).sum(1).mean()),
            "hit_layers_both": float(((n0 > 0) & (n1 > 0)).sum(1).mean()),
            "cold_experts_rank_step": [float(n0.sum(1).mean()), float(n1.sum(1).mean())],
            "per_rank_ms": [float(t0.sum(1).mean()), float(t1.sum(1).mean())],
            "crit_ms": float(np.maximum(t0, t1).sum(1).mean()),
            "cold_route_share": float(cold.mean())}


def calibrate_kappa(p, tier, cost, target_hit_layers, rng, n_tok=3000):
    lo, hi = 0.5, 20.0
    for _ in range(30):
        mid = (lo * hi) ** 0.5
        ids, _ = gen_routes(p, tier, mid, n_tok, rng)
        h = np.mean(base_stats(ids, tier, cost)["hit_layers_rank"])
        if h < target_hit_layers:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def trace_replay_n1(n: np.ndarray, cost: Cost) -> dict:
    """N1 / N1s on the MEASURED per-step, per-layer, per-rank Grace expert counts (R3.1 trace), assuming every
    cold pick is replicated (full replication: ~115 GiB/host). No expert ids survive, so partial budgets can't
    be replayed; this is the upper bound of N1 on real routes."""
    out = {}
    for mode in ("base", "n1", "n1s"):
        tot = []
        for s in range(n.shape[0]):
            c = 0.0
            for l in range(n.shape[1]):
                picks = [(0, True)] * int(n[s, l, 0]) + [(1, True)] * int(n[s, l, 1])
                c += layer_time(cost, picks, mode=mode)[0] if picks else cost.EMPTY
            tot.append(c)
        out[mode] = float(np.mean(tot))
    return out


# ---------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "out" / "n1n2-results.json"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    here = pathlib.Path(__file__).parent
    res: dict = {"provenance": "SYNTHETIC routes calibrated to measured stats unless key starts with 'trace'"}

    cfit = fit_cost([here / "data/e3a16c1-perstep.json", here / "data/g3e2-perstep.json"])
    res["cost_fit"] = cfit
    cost = Cost(cfit["A"], cfit["B"], cfit["EMPTY"])
    tr = trace_stats(here / "data/e3a16c1-perstep.json", cfit)
    n_meas = tr.pop("n")
    res["trace_e3a16c1"] = tr
    STEP_NSYS, TOKS_BENCH = 22.59, 43.9
    F = STEP_NSYS - tr["crit_mean"]
    res["F_ms"] = F
    res["trace_replay_n1_full_replication"] = trace_replay_n1(n_meas, cost)
    res["trace_replay_n1s_half_eff_0.75"] = trace_replay_n1(n_meas, Cost(cost.A, cost.B, cost.EMPTY, 0.75))

    cnt = load_counts()
    p_eval = 0.5 * cnt["acs_h2"] + 0.5 * cnt["gsm"]
    p_sel = 0.7 * cnt["acs_h1"] + 0.3 * cnt["gsm"]
    tier = load_placement()
    target = float(np.mean(tr["hit_layers_rank"]))
    kappa = calibrate_kappa(p_eval, tier, cost, target, rng)
    res["kappa_trace"] = kappa
    S = 800 if a.quick else a.steps

    scen = {"trace-calibrated": kappa, "as-designed (kappa=1, 3.7% Grace routes)": 1.0}
    res["scenarios"] = {}
    for name, kap in scen.items():
        ids, g = gen_routes(p_eval, tier, kap, S, rng)
        sc = {"kappa": kap, "base": base_stats(ids, tier, cost)}
        b = simulate(ids, g, tier, np.zeros_like(tier, bool), cost, "base")
        sc["base_sim"] = b
        crit0 = b["crit_mean"]

        def tok(crit_new, extra=0.0):
            return TOKS_BENCH * (F + crit0) / (F + crit_new + extra)
        # N1 / N1s across replica budgets
        sc["n1"] = []
        for bud in (25, 50, 75, 100, 116):
            rep, cap = replica_sets(p_sel, tier, bud, kap)
            row = {"budget_gib": bud, **cap}
            for mode in ("n1", "n1s"):
                r = simulate(ids, g, tier, rep, cost, mode)
                row[mode + "_crit"] = r["crit_mean"]
                row[mode + "_tok_s"] = tok(r["crit_mean"], 0.18)
            r = simulate(ids, g, tier, rep, Cost(cost.A, cost.B, cost.EMPTY, 0.75), "n1s")
            row["n1s_half_eff0.75_crit"] = r["crit_mean"]
            sc["n1"].append(row)
        sc["n1h"] = []
        for bud in (15, 29, 44, 58):
            rep, cap = replica_sets(p_sel, tier, bud, kap, frac=0.5)
            row = {"budget_gib": bud, **cap}
            for he in (1.0, 0.75):
                r = simulate(ids, g, tier, rep, Cost(cost.A, cost.B, cost.EMPTY, he), "n1h")
                row[f"n1h_eff{he}_crit"] = r["crit_mean"]
                row[f"n1h_eff{he}_tok_s"] = tok(r["crit_mean"], 0.18)
            sc["n1h"].append(row)
        # N2 (alone) and N1+N2 across eps, rho
        sc["n2"] = []
        rep100, _ = replica_sets(p_sel, tier, 100, kap)
        rho_eps = [(0.1, e) for e in (0.01, 0.02, 0.05, 0.1, 0.2)]
        if not a.quick:      # degeneracy check: results depend on ~eps/rho (margin in router-noise units)
            rho_eps += [(0.2, 0.1), (0.05, 0.025)]
        for rho, eps in rho_eps:
            if True:
                for mode, rep in (("n2", np.zeros_like(tier, bool)), ("n1+n2", rep100), ("n1s+n2", rep100)):
                    if a.quick and mode == "n1+n2":
                        continue
                    for budget in ((None,) if mode != "n2" or a.quick or rho != 0.1 else (None, 8)):
                        r = simulate(ids, g, tier, rep, cost, mode, eps=eps, rho=rho, budget=budget)
                        extra = 0.12 + (0.18 if mode != "n2" else 0.0)
                        r.update(rho=rho, eps=eps, mode=mode, budget=budget, tok_s=tok(r["crit_mean"], extra),
                                 removed_frac_of_hit_layers=r["stall_layers_removed_per_step"] / b["hit_layers_any"])
                        sc["n2"].append(r)
        # C16 and DSpark-verify (T=6) sensitivity: Grace crit only, N1 at 100 GiB
        if not a.quick:
            sc["multi_token"] = {}
            for T in (6, 16):
                St = max(200, S // (4 * T // 2))
                idsT, gT = gen_routes(p_eval, tier, kap, St * T, rng)
                bT = simulate(idsT, gT, tier, np.zeros_like(tier, bool), cost, "base", tok_per_step=T)
                n1T = simulate(idsT, gT, tier, rep100, cost, "n1", tok_per_step=T)
                n1sT = simulate(idsT, gT, tier, rep100, cost, "n1s", tok_per_step=T)
                sc["multi_token"][T] = {"base_crit": bT["crit_mean"], "n1_crit": n1T["crit_mean"],
                                        "n1s_crit": n1sT["crit_mean"], "steps": St}
        res["scenarios"][name] = sc
        print(name, "kappa %.2f" % kap, json.dumps(sc["base"]), flush=True)
    pathlib.Path(a.out).parent.mkdir(exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
