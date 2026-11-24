#!/usr/bin/env python3
"""L6-4g Part 3 (Amendment 2, "Statistic and readings"): paired GLM - Hy4 comparison over tasks.

  python3 -I tools/hy4/part3/analyze.py --glm GNC_DIR/bench/graded.jsonl --hy4 HAC_DIR/bench/graded.jsonl \
      [--out results.json]

Per task, pass rate = passes / 4 for each model. Primary: mean over tasks of (GLM - Hy4) pass rate, 95% interval
by paired bootstrap over tasks (10,000 resamples, seed 20261123), exact sign-flip permutation p-value (two-sided,
over tasks with a non-zero difference). Reading: "GLM higher" / "Hy4 higher" if the interval excludes zero,
otherwise "no measurable difference". Gate failures above 5% of a model's replies make Part 3 "not comparable"
for that load. Realised MDD = (1.959964 + 0.841621) * sd(diff) / sqrt(n). Secondary (descriptive): tiers,
pass@4, verdict counts, mean reply length.
"""
import argparse, json, random, statistics as stt
from collections import Counter, defaultdict

SEED, B = 20261123, 10000


def load(p):
    rows = [json.loads(x) for x in open(p)]
    by = defaultdict(list)
    for r in rows:
        by[r['id']].append(r)
    return rows, by


def signflip_p(d):
    """Exact two-sided sign-flip p over tasks. Differences are multiples of 1/4, so the null distribution of the
    sum is counted exactly by dynamic programming over integer quarters (no enumeration limit)."""
    k = [round(4 * x) for x in d if round(4 * x) != 0]
    if not k:
        return 1.0, 'exact'
    obs = abs(sum(k))
    dist = Counter({0: 1})
    for v in k:
        nxt = Counter()
        for s, c in dist.items():
            nxt[s + v] += c
            nxt[s - v] += c
        dist = nxt
    return sum(c for s, c in dist.items() if abs(s) >= obs) / 2 ** len(k), 'exact'


def compare(tasks, pg, ph):
    d = [pg[t] - ph[t] for t in tasks]
    n = len(d)
    mean = sum(d) / n
    rng = random.Random(SEED)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    lo, hi = boots[int(0.025 * B)], boots[int(0.975 * B) - 1]
    p, kind = signflip_p(d)
    sd = stt.stdev(d) if n > 1 else 0.0
    reading = 'GLM higher' if lo > 0 else 'Hy4 higher' if hi < 0 else 'no measurable difference'
    return dict(n_tasks=n, mean_diff=mean, ci95=[lo, hi], p_signflip=p, p_kind=kind, reading=reading,
                mdd=(1.959964 + 0.841621) * sd / n ** .5, glm_mean=sum(pg[t] for t in tasks) / n,
                hy4_mean=sum(ph[t] for t in tasks) / n,
                tasks_glm_better=sum(x > 0 for x in d), tasks_hy4_better=sum(x < 0 for x in d))


def describe(rows, by):
    toks = [r['completion_tokens'] for r in rows if r.get('completion_tokens') is not None]
    return dict(replies=len(rows), passed=sum(r['passed'] for r in rows), verdicts=Counter(r['verdict'] for r in rows),
                gate_failures=sum(not r['gate'] for r in rows),
                gate_failure_rate=sum(not r['gate'] for r in rows) / len(rows),
                pass_at_4=sum(any(r['passed'] for r in v) for v in by.values()),
                mean_completion_tokens=sum(toks) / len(toks) if toks else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glm', required=True)
    ap.add_argument('--hy4', required=True)
    ap.add_argument('--out')
    a = ap.parse_args()
    rg, bg = load(a.glm)
    rh, bh = load(a.hy4)
    tasks = sorted(set(bg) | set(bh))
    for t in tasks:
        assert len(bg[t]) == 4 and len(bh[t]) == 4, f'{t}: need 4 graded samples per model'
    tier = {t: bg[t][0]['tier'] for t in tasks}
    pg = {t: sum(r['passed'] for r in bg[t]) / 4 for t in tasks}
    ph = {t: sum(r['passed'] for r in bh[t]) / 4 for t in tasks}
    dg, dh = describe(rg, bg), describe(rh, bh)
    res = dict(glm=dg, hy4=dh, primary=compare(tasks, pg, ph))
    bad = [m for m, x in (('GLM', dg), ('Hy4', dh)) if x['gate_failure_rate'] > 0.05]
    if bad:
        res['primary']['reading'] = 'not comparable (gate failures > 5%: ' + ', '.join(bad) + ')'
    res['tiers'] = {}
    for k in sorted(set(tier.values())):
        ts = [t for t in tasks if tier[t] == k]
        res['tiers'][k] = dict(n_tasks=len(ts), glm_mean=sum(pg[t] for t in ts) / len(ts),
                               hy4_mean=sum(ph[t] for t in ts) / len(ts),
                               glm_pass_at_4=sum(any(r['passed'] for r in bg[t]) for t in ts),
                               hy4_pass_at_4=sum(any(r['passed'] for r in bh[t]) for t in ts))
    res['per_task'] = {t: dict(tier=tier[t], glm=pg[t], hy4=ph[t]) for t in tasks}
    out = json.dumps(res, indent=1, default=dict)
    print(out)
    if a.out:
        open(a.out, 'w').write(out + '\n')


if __name__ == '__main__':
    main()
