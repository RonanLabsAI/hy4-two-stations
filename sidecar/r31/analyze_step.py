"""analyze_step.py -- R3.1: break a DS-V4-Pro decode step (nsys sqlite, one rank) into kernel categories.

usage: python3 analyze_step.py trace.sqlite [--marker REGEX] [--names] [--json out.json]

Steps: consecutive occurrences of a once-per-iteration marker kernel (--marker; default: auto = the kernel name whose
count is closest to the iteration count among names that occur 20-60 times and sits in the largest pre-gap). Steps
whose length is > 1.5x the median are dropped (prefill). Per step and category: busy = sum of kernel durations,
plus 'gpu_idle' = step minus the union of all kernel intervals (launch gaps / host work / waits outside kernels).
E3 sidecar: the _wait kernel is the exposed sidecar time; _pack/_publish/_scatter_add are the GB300-side overheads.
"""
import argparse, json, re, sqlite3, statistics as st
from collections import Counter, defaultdict

CATS = [  # first match wins; matched against the short and demangled names
    ("e3_wait", r"^_wait$|_wait\b"),
    ("e3_other", r"^_pack$|^_publish$|^_scatter_add$|^_globaltimer$"),
    ("nccl", r"nccl"),
    ("marlin", r"[Mm]arlin"),
    ("moe_route", r"moe_align|moe_sum|topk|top_k|router|gating|grouped_topk|count_and_sort|expert_map"),
    ("attn_dsa", r"flash_?mla|sparse|indexer|compress|paged|attn|attention|swa|fmha|mla|kv_cache|cache_kernel|hadamard"),
    ("mhc", r"mhc|hc_head|hc_pre|hc_post|sinkhorn"),
    ("dense_gemm", r"gemm|cutlass|nvjet|cublas|sm1\d\d_|Gemm|matmul|deep_gemm|fp8_mqa|bmm"),
    ("norm_quant_elem", r"norm|rope|rotary|quant|silu|act_and_mul|elementwise|vectorized|reduce|copy|fill|cast|cat|index|scatter|gather"),
    ("sampler_spec", r"sample|argmax|softmax|logit|rejection|spec|draft|penalt"),
]


def load(path):
    db = sqlite3.connect(path)
    S = dict(db.execute("select id, value from StringIds"))
    cols = [r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
    nm = "shortName" if "shortName" in cols else "demangledName"
    return [(s, e, S.get(n, str(n)), S.get(dn, str(dn)), stream) for s, e, n, dn, stream in
            db.execute(f"select start,end,{nm},demangledName,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]


def cat_of(short, dem, rx):
    for c, r in rx:
        if r.search(short) or r.search(dem[:300]):
            return c
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite"); ap.add_argument("--marker", default=""); ap.add_argument("--names", action="store_true")
    ap.add_argument("--json", default=""); ap.add_argument("--maxx", type=float, default=1.5)
    a = ap.parse_args()
    K = load(a.sqlite)
    rx = [(c, re.compile(r)) for c, r in CATS]
    cnt = Counter(k[2] for k in K)
    if a.names:
        tot = defaultdict(int)
        for s, e, n, d, _ in K:
            tot[n] += e - s
        for n, t in sorted(tot.items(), key=lambda x: -x[1])[:60]:
            print(f"{t/1e6:9.2f} ms {cnt[n]:7d}  {cat_of(n, '', rx):15s} {n[:110]}")
        print("candidates (count 15-80):", [(n[:60], c) for n, c in cnt.items() if 15 <= c <= 80][:40])
    if a.marker:
        mrx = re.compile(a.marker)
        marks = [k[0] for k in K if mrx.search(k[2])]
    else:
        return
    steps = []
    for i in range(len(marks) - 1):
        steps.append((marks[i], marks[i + 1]))
    lens = [(b - x) / 1e6 for x, b in steps]
    med = st.median(lens)
    keep = [s for s, l in zip(steps, lens) if l <= a.maxx * med]
    per = []
    idx = 0
    for s0, s1 in keep:
        busy = defaultdict(float); ncall = Counter(); union = 0.0; cur_s = cur_e = None
        names = defaultdict(float)
        for s, e, n, d, _ in K:
            if s < s0 or s >= s1:
                continue
            e2 = min(e, s1)
            c = cat_of(n, d, rx)
            busy[c] += (e2 - s) / 1e6; ncall[c] += 1; names[(c, n)] += (e2 - s) / 1e6
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    union += (cur_e - cur_s) / 1e6
                cur_s, cur_e = s, e2
            else:
                cur_e = max(cur_e, e2)
        if cur_e is not None:
            union += (cur_e - cur_s) / 1e6
        step = (s1 - s0) / 1e6
        per.append({"step_ms": step, "busy": dict(busy), "calls": dict(ncall), "gpu_idle_ms": step - union,
                    "names": {f"{c}|{n[:80]}": v for (c, n), v in names.items()}})
    def m(f):
        return st.median(f(p) for p in per)
    cats = sorted({c for p in per for c in p["busy"]})
    res = {"steps": len(per), "dropped": len(steps) - len(keep), "step_ms": m(lambda p: p["step_ms"]),
           "gpu_idle_ms": m(lambda p: p["gpu_idle_ms"]),
           "busy_ms": {c: round(m(lambda p, c=c: p["busy"].get(c, 0.0)), 4) for c in cats},
           "calls": {c: m(lambda p, c=c: p["calls"].get(c, 0)) for c in cats}}
    allnames = sorted({k for p in per for k in p["names"]})
    res["top_names"] = sorted(((k, round(m(lambda p, k=k: p["names"].get(k, 0.0)), 4)) for k in allnames), key=lambda x: -x[1])[:40]
    res["step_lens_ms"] = [round(x, 3) for x in lens]
    print(json.dumps({k: v for k, v in res.items() if k not in ("top_names",)}, indent=1))
    for k, v in res["top_names"]:
        print(f"  {v:8.4f}  {k}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
