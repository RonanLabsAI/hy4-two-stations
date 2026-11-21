# ruff: noqa: E741
"""Extract per-step, per-layer, per-rank cold (Grace) marlin time from the R3.1
E3-A16 C1 nsys sqlite pair (<tag>-r{0,1}.sqlite; tag e3a16c1 = R3.1 E3-A16 real text, g3e2 = R3.3 E2 catid). Timing only, no text/routes.

Step delimiter and marlin grouping follow r31/detail.py: 4 Marlin launches per
MoE layer, [0:2] = hot (HBM) w13/w2, [2:4] = cold (Grace, E2 split) w13/w2.
Output: data/e3a16c1-perstep.json  {rank: {"cold": [[61 ms]...steps], "hot": ..., "step_ms": [...]}}
"""
import json
import pathlib
import sqlite3
import statistics as st
import sys

src = pathlib.Path(sys.argv[1])
tag = sys.argv[2] if len(sys.argv) > 2 else "e3a16c1"
out = {}
for r in (0, 1):
    db = sqlite3.connect(src / f"{tag}-r{r}.sqlite")
    S = dict(db.execute("select id,value from StringIds"))
    K = [(s, e, S[n]) for s, e, n in db.execute(
        "select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    marks = [s for s, e, n in K if n == "vocab_parallel_embedding_kernel"]
    steps = [(marks[i], marks[i + 1]) for i in range(len(marks) - 1)]
    L = [(b - a) / 1e6 for a, b in steps]
    m = st.median(L)
    steps = [s for s, l in zip(steps, L) if l < 1.5 * m]
    cold, hot, lens, nmk = [], [], [], []
    for a, b in steps:
        mk = [k for k in K if a <= k[0] < b and k[2] == "Marlin"]
        nmk.append(len(mk))
        g = [mk[i:i + 4] for i in range(0, len(mk), 4)][:61]
        hot.append([round(sum(x[1] - x[0] for x in G[:2]) / 1e6, 5) for G in g])
        cold.append([round(sum(x[1] - x[0] for x in G[2:]) / 1e6, 5) for G in g])
        lens.append(round((b - a) / 1e6, 4))
    out[r] = dict(cold=cold, hot=hot, step_ms=lens, marlin_per_step=nmk)
    print(r, "steps", len(steps), "marlin/step", set(nmk))
dst = pathlib.Path(__file__).parent / "data" / f"{tag}-perstep.json"
json.dump(out, open(dst, "w"))
print("wrote", dst)
