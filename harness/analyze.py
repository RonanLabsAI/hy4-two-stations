#!/usr/bin/env python3
"""Offline L6-4h analysis. Stdlib only.

  drift   REF_DIR TEST_DIR   top-k logprob drift of TEST against REF (the FP8-B reference) at identical positions
  paired  A_DIR B_DIR        paired quality (same item IDs): accuracy difference, Newcombe 95% CI, exact McNemar
  abba    FILE.json          A-B-B-A speed confirm: paired log-ratio Student t95 from {"A":[...],"B":[...]} pairs

Drift estimator (frozen). At each probe position the reference gives its top-k {id: logprob}; the test gives its
own top-k. Distributions are coarse-grained to the reference's top-k ids plus one tail bucket:
  p_i = exp(ref_i);            p_tail = max(eps, 1 - sum p_i)
  q_i = exp(test_i) if i in test top-k else min(exp(min test), test tail mass)   (the largest q the test allows)
  q_tail = max(eps, 1 - sum q_i)
  KL_k = sum_i p_i log(p_i/q_i) + p_tail log(p_tail/q_tail)
By the data-processing inequality, coarse-graining can only lower KL, and the q chosen for missing ids is the
largest possible, so KL_k is a conservative (low) estimate of full-vocabulary KL(ref || test). Also reported:
top-1 agreement, rank of the reference argmax in the test list (k+1 if absent), and the per-token NLL of the
true corpus token under each model (floored at log of the smallest listed probability, flagged).
Uncertainty: cluster bootstrap over windows (10,000 resamples, seed 42), percentile 95%.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

EPS = 1e-10


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def kl_bucketed(ref, test):
    p = {i: math.exp(v) for i, v in ref.items()}
    tq = {i: math.exp(v) for i, v in test.items()}
    t_tail = max(0.0, 1-sum(tq.values()))
    q_missing = min(min(tq.values()), t_tail) if tq else EPS
    q = {i: tq.get(i, q_missing) for i in p}
    p_tail = max(EPS, 1-sum(p.values()))
    q_tail = max(EPS, 1-sum(q.values()))
    kl = sum(pi*math.log(pi/max(q[i], EPS)) for i, pi in p.items() if pi > 0)
    return kl+p_tail*math.log(p_tail/q_tail)


def position_metrics(r, t, k):
    ref = {int(i): v for i, v in r['top']}
    test = {int(i): v for i, v in t['top']}
    ref_top = max(ref, key=ref.get)
    order = sorted(test, key=lambda i: -test[i])
    rank = order.index(ref_top)+1 if ref_top in test else k+1

    def nll(d):
        if r['target'] in d:
            return -d[r['target']], False
        return -min(d.values()), True
    nr, fr = nll(ref); nt, ft = nll(test)
    return dict(kl=kl_bucketed(ref, test), top1=ref_top == max(test, key=test.get), rank=rank,
                nll_ref=nr, nll_test=nt, floored=fr or ft, mass_ref=sum(math.exp(v) for v in ref.values()))


def bootstrap(groups, stat, n=10000, seed=42):
    rng = random.Random(seed)
    keys = sorted(groups)
    vals = []
    for _ in range(n):
        sample = [groups[rng.choice(keys)] for _ in keys]
        vals.append(stat([x for g in sample for x in g]))
    vals.sort()
    return [vals[int(.025*n)], vals[int(.975*n)-1]]


def mean(x):
    return sum(x)/len(x) if x else float('nan')


def drift(ref_dir, test_dir, k=64):
    ref = {(x['window'], x['pos']): x for x in read_jsonl(Path(ref_dir)/'logprobs.jsonl')}
    test = {(x['window'], x['pos']): x for x in read_jsonl(Path(test_dir)/'logprobs.jsonl')}
    keys = sorted(set(ref) & set(test))
    missing = sorted(set(ref) ^ set(test))
    bad = [key for key in keys if not (ref[key]['gate'] and test[key]['gate'])]
    target_mismatch = [key for key in keys if ref[key]['target'] != test[key]['target']]
    out = {'positions': len(keys), 'unmatched_positions': len(missing), 'gate_failures': len(bad),
           'target_mismatches': len(target_mismatch)}
    if target_mismatch:
        out['error'] = 'reference and test were run on different corpora'
        return out
    rows = {key: position_metrics(ref[key], test[key], k) for key in keys if key not in bad}
    for kind in ('all', 'text', 'code'):
        sel = {key: v for key, v in rows.items() if kind == 'all' or ref[key]['kind'] == kind}
        groups = {}
        for (w, _), v in sel.items():
            groups.setdefault(w, []).append(v)
        res = {}
        for name, f in (('mean_kl', lambda xs: mean([x['kl'] for x in xs])),
                        ('top1_agreement', lambda xs: mean([x['top1'] for x in xs])),
                        ('delta_nll', lambda xs: mean([x['nll_test']-x['nll_ref'] for x in xs]))):
            res[name] = f([v for g in groups.values() for v in g])
            res[name+'_ci95'] = bootstrap(groups, f) if groups else None
        kls = sorted(v['kl'] for v in sel.values())
        res['p99_kl'] = kls[int(.99*(len(kls)-1))] if kls else None
        res['median_kl'] = kls[len(kls)//2] if kls else None
        res['floored_nll_share'] = mean([v['floored'] for v in sel.values()])
        res['mean_ref_topk_mass'] = mean([v['mass_ref'] for v in sel.values()])
        res['positions'] = len(sel)
        out[kind] = res
    return out


def paired(a_dir, b_dir, set_mode):
    a = {x['id']: x for x in read_jsonl(Path(a_dir)/'quality.jsonl') if f"{x['set']}:{x['mode']}" == set_mode}
    b = {x['id']: x for x in read_jsonl(Path(b_dir)/'quality.jsonl') if f"{x['set']}:{x['mode']}" == set_mode}
    ids = sorted(set(a) & set(b))
    n10 = sum(a[i]['correct'] and not b[i]['correct'] for i in ids)
    n01 = sum(b[i]['correct'] and not a[i]['correct'] for i in ids)
    n11 = sum(a[i]['correct'] and b[i]['correct'] for i in ids)
    n00 = len(ids)-n10-n01-n11
    diff = (n01-n10)/len(ids) if ids else None
    return dict(n=len(ids), a_acc=(n10+n11)/len(ids), b_acc=(n01+n11)/len(ids), diff_b_minus_a=diff,
                newcombe95=newcombe_paired(n11, n10, n01, n00), mcnemar_exact_p=mcnemar(n10, n01),
                discordant=dict(a_only=n10, b_only=n01))


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 1.0)
    p = k/n; d = 1+z*z/n; c = p+z*z/(2*n); h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)


def newcombe_paired(a, b, c, d):
    """Newcombe (1998) method 10 interval for a paired difference of proportions (B - A).
    a = both right, b = A only, c = B only, d = both wrong."""
    n = a+b+c+d
    if n == 0:
        return None
    p1 = (a+b)/n; p2 = (a+c)/n
    l1, u1 = wilson(a+b, n); l2, u2 = wilson(a+c, n)
    dlt = p2-p1
    denom = (a+b)*(c+d)*(a+c)*(b+d)
    phi = 0.0
    if denom > 0:
        num = a*d-b*c
        num = max(0.0, num-n/2) if num > 0 else num
        phi = num/math.sqrt(denom)
    lo = dlt-math.sqrt((p2-l2)**2-2*phi*(p2-l2)*(u1-p1)+(u1-p1)**2)
    hi = dlt+math.sqrt((u2-p2)**2-2*phi*(u2-p2)*(p1-l1)+(p1-l1)**2)
    return [lo, hi]


def mcnemar(b, c):
    n = b+c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k+1))/2**n
    return min(1.0, 2*tail)


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262157, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 19: 2.093}


def abba(pairs):
    """pairs: list of [a, b] throughput values; returns mean ratio and back-transformed t95 interval."""
    logs = [math.log(b/a) for a, b in pairs]
    n = len(logs); mu = mean(logs)
    sd = math.sqrt(sum((x-mu)**2 for x in logs)/(n-1))
    t = T95[n-1]
    h = t*sd/math.sqrt(n)
    return dict(n=n, ratio=math.exp(mu), ci95=[math.exp(mu-h), math.exp(mu+h)], df=n-1, t_crit=t)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('drift'); d.add_argument('ref'); d.add_argument('test'); d.add_argument('--topk', type=int, default=64)
    p = sub.add_parser('paired'); p.add_argument('a'); p.add_argument('b'); p.add_argument('--set', default='gsm8k:no_think')
    b = sub.add_parser('abba'); b.add_argument('file')
    args = ap.parse_args(argv)
    if args.cmd == 'drift':
        res = drift(args.ref, args.test, args.topk)
    elif args.cmd == 'paired':
        res = paired(args.a, args.b, args.set)
    else:
        res = abba(json.loads(Path(args.file).read_text())['pairs'])
    print(json.dumps(res, indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
