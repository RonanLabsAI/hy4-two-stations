#!/usr/bin/env python3
"""L6-4g Part 2 rescore under Amendment 1 as written: a reply that hits the cap without closing its think block
is wrong. The run-time scorer (hy4bench.py answer_text) fell back to scanning the reasoning text when no closing
tag existed, so 44 G-N and 28 H-A capped replies were counted correct. Raw files are not changed.

  python3 -I tools/hy4/part2_rescore.py --glm <gnt>/bench/quality-think.jsonl --hy4 <hat>/bench/quality-think.jsonl
"""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


hb = mod('hb', HERE/'hy4bench.py')
an = mod('an', HERE/'analyze.py')


def rescore(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        closed = hb.has_think_close(r['text'])
        out[r['id']] = dict(ok=bool(r['correct']) and (closed or not r['hit_cap']), cap=r['hit_cap'],
                            open_cap_counted=r['hit_cap'] and not closed and bool(r['correct']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glm', required=True)
    ap.add_argument('--hy4', required=True)
    ap.add_argument('--out')
    a = ap.parse_args()
    g, h = rescore(a.glm), rescore(a.hy4)
    assert sorted(g) == sorted(h)
    ids = sorted(g); n = len(ids)
    n11 = sum(g[i]['ok'] and h[i]['ok'] for i in ids)
    n10 = sum(g[i]['ok'] and not h[i]['ok'] for i in ids)
    n01 = sum(h[i]['ok'] and not g[i]['ok'] for i in ids)
    n00 = n-n11-n10-n01
    ci = an.newcombe_paired(n11, n10, n01, n00)  # for (hy4 - glm), as the run-time analysis reported G-N - H-A
    nc = [i for i in ids if not g[i]['cap'] and not h[i]['cap']]
    res = dict(n=n, glm_correct=sum(v['ok'] for v in g.values()), hy4_correct=sum(v['ok'] for v in h.values()),
               glm_open_cap_counted_at_runtime=sum(v['open_cap_counted'] for v in g.values()),
               hy4_open_cap_counted_at_runtime=sum(v['open_cap_counted'] for v in h.values()),
               glm_minus_hy4=(n10-n01)/n, glm_minus_hy4_newcombe95=[-ci[1], -ci[0]],
               discordant_glm_only=n10, discordant_hy4_only=n01, mcnemar_exact_p=an.mcnemar(n10, n01),
               neither_capped=dict(n=len(nc), glm=sum(g[i]['ok'] for i in nc), hy4=sum(h[i]['ok'] for i in nc)))
    s = json.dumps(res, indent=1)
    print(s)
    if a.out:
        Path(a.out).write_text(s+'\n')


if __name__ == '__main__':
    main()
