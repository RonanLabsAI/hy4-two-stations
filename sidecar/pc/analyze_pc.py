#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""analyze_pc.py SNAPDIR [--e2 E2ROWMAP --e3 E3ROWMAP] [--out OUT.json] -- PC (2026-10-09) placement check.

Reads the E3 hook's tier-counter snapshots (SNAPDIR/<name>-r{0,1}.json, order from SNAPDIR/index.txt), takes deltas
between consecutive snapshots (one bench run each), and reports per decode step (bucket 0 = T == 1 calls) and per rank:
routes and unique experts per tier (HBM hot / 6000 warm / Grace cold), Grace hit layers, Grace share, and the
cross-rank critical-path Grace time from the per-layer histograms (independence across ranks, N1N2 cost fit
t(n) = 47.6 + 84.8 n us, t(0) = 3 us). Also checks:
  * live tiers vs the rowmap files (the counters' tiers come from the LIVE maps the marlin calls read),
  * the placement's own prediction on the calibration traffic (ACS h2 / GSM decode counts),
  * id space: per-layer cosine of the run's per-expert decode counts vs the calibration counts.
"""
import argparse
import json
import os

import numpy as np

E, L, NL = 384, 61, 192
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
F_MS, T0_US, A_US, B_US = 17.19, 3.0, 47.6, 84.8     # N1N2-SIMULATOR.md section 2
BASE_TOK, BASE_CRIT = 43.9, 5.40


def tcost(n):
    return np.where(n == 0, T0_US, A_US + B_US * n) / 1000.0


def load_snap(d, name):
    out = {}
    for r in (0, 1):
        j = json.load(open(os.path.join(d, f"{name}-r{r}.json")))
        out[r] = j
    return out


def arrs(j, b):
    """-> counts [L,E], routes [L,4], unique [L,4], hist [L,17] for bucket b."""
    t, bb = j["tiers"], j["by_bucket"][str(b)]
    cnt = np.array([bb[str(l)] for l in range(L)], float)
    rt = np.array([t[str(l)][str(b)]["routes"] for l in range(L)], float)
    un = np.array([t[str(l)][str(b)]["unique"] for l in range(L)], float)
    hi = np.array([t[str(l)][str(b)]["hist_unique_cold"] for l in range(L)], float)
    return cnt, rt, un, hi


def tiers_from_rowmaps(e2, e3):
    """code[rank][l, g]: 0 remote 1 hot 2 warm 3 cold."""
    e2l = {int(k): set(v["hot"]) for k, v in json.load(open(e2))["layers"].items()}
    e3l = {int(k): set(v["warm"]) for k, v in json.load(open(e3))["layers"].items()}
    codes = {}
    for r in (0, 1):
        c = np.zeros((L, E), int)
        for l in range(L):
            for g in range(r * NL, r * NL + NL):
                c[l, g] = 1 if g in e2l.get(l, ()) else 2 if g in e3l.get(l, ()) else 3
        codes[r] = c
    return codes


def crit_from_hist(h0, h1):
    """Expected sum over layers of max over ranks of t(n), ranks independent; h* [L,17] counts."""
    n = np.arange(h0.shape[1])
    tot = 0.0
    for l in range(L):
        p0 = h0[l] / max(h0[l].sum(), 1)
        p1 = h1[l] / max(h1[l].sum(), 1)
        t = tcost(n)
        tot += float((p0[:, None] * p1[None, :] * np.maximum(t[:, None], t[None, :])).sum())
    return tot


def calib(path_acs, path_gsm):
    acs = json.load(open(path_acs))["splits"]
    gsm = json.load(open(path_gsm))["phases"]
    def a(layers):
        x = np.zeros((L, E))
        for k, v in layers.items():
            x[int(k)] = v
        return x
    return {"acs-h1": a(acs["acs-h1"]["decode"]), "acs-h2": a(acs["acs-h2"]["decode"]), "gsm": a(gsm["decode"])}


def predicted_from_counts(cnt, codes):
    """Per decode token, per rank: expected routes per tier (sum over layers) given per-layer expert frequencies
    cnt [L,E] (any traffic), and expected cold hit layers / unique cold experts at T == 1 (unique == routes)."""
    out = {}
    tok = cnt.sum(1) / 6.0                  # tokens per layer
    for r in (0, 1):
        res = {}
        for i, name in enumerate(("remote", "hot", "warm", "cold")):
            m = codes[r] == i
            res[name] = float(((cnt * m).sum(1) / np.maximum(tok, 1)).sum())
        loc = res["hot"] + res["warm"] + res["cold"]
        res["grace_share_of_local"] = res["cold"] / max(loc, 1e-9)
        out[r] = res
    return out


def cos_layers(a, b):
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(den, 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("snapdir")
    p.add_argument("--e2", default=os.path.join(ROOT, "calibration/rowmap-dsv4pro-acs-v1.json"))
    p.add_argument("--e3", default=os.path.join(ROOT, "rowmaps/rowmap-e3-on-e2-acs-v1.json"))
    p.add_argument("--acs", default=os.path.join(ROOT, "calibration/routes-e1-acs.json"))
    p.add_argument("--gsm", default=os.path.join(ROOT, "calibration/routes-e1-gsm8k100.json"))
    p.add_argument("--out", default="")
    a = p.parse_args()
    names = [ln.split()[0] for ln in open(os.path.join(a.snapdir, "index.txt")) if ln.strip()]
    snaps = {n: load_snap(a.snapdir, n) for n in names}
    codes = tiers_from_rowmaps(a.e2, a.e3)
    cal = calib(a.acs, a.gsm)
    rep = {"names": names, "tier_info_check": {}, "runs": {}, "calib_prediction": {}, "groups": {}}

    # live tiers vs rowmap files
    first = snaps[names[0]]
    for r in (0, 1):
        ti = first[r]["tier_info"]
        bad = []
        for l in range(L):
            x = ti[str(l)]
            want = {t: int((codes[r][l] == i).sum()) for i, t in enumerate(("remote", "hot", "warm", "cold"))}
            want["remote"] = E - NL
            got = {t: x[t] for t in ("remote", "hot", "warm", "cold")}
            if got != want or x.get("overlap_warm_cold") or x.get("overlap_hot_cold") or x.get("local_unassigned"):
                bad.append((l, got, want, {k: x.get(k) for k in ("overlap_warm_cold", "overlap_hot_cold",
                                                                    "local_unassigned")}))
        tot = {t: sum(ti[str(l)][t] for l in range(L)) for t in ("hot", "warm", "cold")}
        rep["tier_info_check"][r] = {"layers_mismatch": len(bad), "examples": bad[:3], "totals": tot,
                                     "cold_w1_device": ti["0"].get("cold_w1_device")}

    for nm in ("acs-h1", "acs-h2", "gsm"):
        rep["calib_prediction"][nm] = predicted_from_counts(cal[nm], codes)

    prev = names[0]
    for n in names[1:]:
        run = {}
        cnts = {}
        hists = {}
        for r in (0, 1):
            c1, rt1, un1, h1 = arrs(snaps[n][r], 0)
            c0, rt0, un0, h0 = arrs(snaps[prev][r], 0)
            cnt, rt, un, hi = c1 - c0, rt1 - rt0, un1 - un0, h1 - h0
            steps = hi.sum(1)
            S = float(np.median(steps))
            cnts[r], hists[r] = cnt, hi
            # file-based tiers applied to the measured per-expert counts (must equal live-tier routes)
            filetier = [float((cnt * (codes[r] == i)).sum()) for i in range(4)]
            run[r] = {"decode_steps": S, "steps_min_max": [float(steps.min()), float(steps.max())],
                      "routes_per_step": dict(zip(("remote", "hot", "warm", "cold"), (rt.sum(0) / max(S, 1)).round(3).tolist())),
                      "unique_per_step": dict(zip(("remote", "hot", "warm", "cold"), (un.sum(0) / max(S, 1)).round(3).tolist())),
                      "grace_share_of_local_routes": round(float(rt[:, 3].sum() / max(rt[:, 1:].sum(), 1)), 4),
                      "grace_hit_layers_per_step": round(float((hi[:, 1:].sum(1) / np.maximum(steps, 1)).sum()), 2),
                      "rowmap_file_tier_routes_per_step": dict(zip(("remote", "hot", "warm", "cold"),
                                                                    (np.array(filetier) / max(S, 1)).round(3).tolist())),
                      "cold_per_layer_per_step": (rt[:, 3] / np.maximum(steps, 1)).round(3).tolist()}
        if run[0]["decode_steps"] > 0 and run[1]["decode_steps"] > 0:
            crit = crit_from_hist(hists[0], hists[1])
            per = [sum(float((hists[r][l] / max(hists[r][l].sum(), 1) * tcost(np.arange(17))).sum()) for l in range(L))
                   for r in (0, 1)]
            both = float(sum((hists[0][l, 1:].sum() / max(hists[0][l].sum(), 1)) * (hists[1][l, 1:].sum() / max(hists[1][l].sum(), 1)) for l in range(L)))
            run["crit_grace_ms"] = round(crit, 3)
            run["per_rank_grace_ms"] = [round(x, 3) for x in per]
            run["both_rank_hit_layers"] = round(both, 2)
            run["pred_c1_from_crit"] = round(BASE_TOK * (F_MS + BASE_CRIT) / (F_MS + crit), 1)
            tot = cnts[0]  # both ranks see all routes (EP2, all tokens on both)
            run["idspace_cos_vs"] = {k: round(float(np.median(cos_layers(tot, v))), 3) for k, v in cal.items()}
            run["routes_check_ranks_equal"] = bool(np.allclose(cnts[0], cnts[1]))
        rep["runs"][n] = run
        prev = n
    # reference id-space cosines between calibration sets
    rep["idspace_cos_calib"] = {"acs-h1_vs_acs-h2": round(float(np.median(cos_layers(cal["acs-h1"], cal["acs-h2"]))), 3),
                                "acs-h2_vs_gsm": round(float(np.median(cos_layers(cal["acs-h2"], cal["gsm"]))), 3),
                                "acs-h2_vs_uniform": round(float(np.median(cos_layers(cal["acs-h2"], np.ones((L, E))))), 3)}
    s = json.dumps(rep, indent=1)
    if a.out:
        open(a.out, "w").write(s)
    # short text summary
    print("tier check:", {r: (v["layers_mismatch"], v["totals"]) for r, v in rep["tier_info_check"].items()})
    for nm, v in rep["calib_prediction"].items():
        print(f"calib {nm}: cold/token-step r0 {v[0]['cold']:.2f} r1 {v[1]['cold']:.2f}  share {v[0]['grace_share_of_local']:.3f}/{v[1]['grace_share_of_local']:.3f}")
    print("idspace calib:", rep["idspace_cos_calib"])
    for n, run in rep["runs"].items():
        if 0 not in run:
            continue
        r0, r1 = run[0], run[1]
        print(f"{n:10s} steps {r0['decode_steps']:.0f} cold/step {r0['routes_per_step']['cold']:.2f}/{r1['routes_per_step']['cold']:.2f} "
              f"warm {r0['routes_per_step']['warm']:.2f}/{r1['routes_per_step']['warm']:.2f} hot {r0['routes_per_step']['hot']:.1f}/{r1['routes_per_step']['hot']:.1f} "
              f"share {r0['grace_share_of_local_routes']:.3f}/{r1['grace_share_of_local_routes']:.3f} hitL {r0['grace_hit_layers_per_step']}/{r1['grace_hit_layers_per_step']} "
              f"crit {run.get('crit_grace_ms')} both {run.get('both_rank_hit_layers')} cos {run.get('idspace_cos_vs')} filecold {r0['rowmap_file_tier_routes_per_step']['cold']}")


if __name__ == "__main__":
    main()
