#!/usr/bin/env python3
"""L6-4g Part 3 (Amendment 2, "Grading"): grade one load's code.jsonl against the frozen tests. CPU only.

  python3 -I tools/hy4/part3/grade.py RUN_DIR/bench/code.jsonl --out RUN_DIR/bench/graded.jsonl [--workers 16]

Rule (frozen): take the last ```cpp block in the text after the final closing think tag; compile with g++ 13
-std=gnu++20 -O2; run every frozen test of the task in the bwrap sandbox with the task's time and memory limits.
A sample passes only if every test is correct. No code block, compile error, cap hit, or any failed test is a
fail, counted by type. Replies with a code block in another fence (```c++, ```cc, ```) are fails and also
counted as `other_fence` (descriptive only).

Before grading, every test file is re-hashed against BUILD.json, and the set against data/part3/SET.json.
Grade on a quiet machine: limits are CPU time, but heavy load still inflates it. The grader refuses to start
above --max-load (default 32 on the 64-core Threadripper), and a TLE counts only if it repeats on an immediate
re-run of that test (fixed 2026-11-23, before any Part 3 output existed).
"""
import argparse, concurrent.futures as cf, hashlib, json, os, re, sys, tempfile, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import sandbox as sb  # noqa: E402
import hy4bench  # noqa: E402

DATA = HERE.parent / 'data' / 'part3'
CPP_RE = re.compile(r'```cpp[ \t]*\n(.*?)```', re.S)
OTHER_RE = re.compile(r'```(?:c\+\+|cc|cxx|C\+\+|CPP)?[ \t]*\n(.*?)```', re.S)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_freeze():
    st = json.loads((DATA / 'SET.json').read_text())
    tasks = {}
    for t in st['tasks']:
        d = DATA / 'tasks' / t['task_id']
        b = json.loads((d / 'BUILD.json').read_text())
        if b['status'] != 'ok' or sha(d / 'BUILD.json') != t['build_sha256']:
            sys.exit(f"{t['task_id']}: BUILD.json not ok or hash differs from SET.json")
        for name, h in b['tests'].items():
            if sha(d / 'tests' / name) != h:
                sys.exit(f"{t['task_id']}: test {name} hash mismatch (regenerate with build_tests.py and recheck)")
        chk = None
        if b.get('checker'):
            chk = Path(tempfile.mkdtemp(prefix='p3chk-')) / 'checker'
            ok, err = sb.compile_cpp(d / 'checker.cpp', chk)
            if not ok:
                sys.exit(f"{t['task_id']}: checker compile failed")
        ins = sorted((d / 'tests').glob('*.in'))
        tasks[t['task_id']] = dict(tl=b['time_limit_s'], ml=b['memory_limit_mib'], checker=chk, tests=ins)
    return st, tasks


def extract(text):
    body = hy4bench.answer_text(text)
    blocks = CPP_RE.findall(body)
    if blocks:
        return blocks[-1], None
    return None, 'other_fence' if OTHER_RE.search(body) else 'no_code'


def grade_one(row, tasks):
    t = tasks[row['id']]
    base = dict(id=row['id'], tier=row['tier'], seed=row['seed'], completion_tokens=row.get('completion_tokens'))
    gate = bool(row.get('gate_prompt')) and bool(row.get('think_gate'))
    if row.get('error'):
        return dict(base, passed=False, verdict='request_error', gate=False)
    if row.get('hit_cap'):
        return dict(base, passed=False, verdict='cap_hit', gate=gate)
    code, why = extract(row['text'])
    if code is None:
        return dict(base, passed=False, verdict=why, gate=gate)
    work = Path(tempfile.mkdtemp(prefix='p3g-'))
    try:
        (work / 'main.cpp').write_text(code)
        ok, err = sb.compile_cpp(work / 'main.cpp', work / 'main')
        if not ok:
            return dict(base, passed=False, verdict='compile_error', gate=gate, detail=err[-500:])
        max_cpu = 0.0
        for inp in t['tests']:
            r = sb.run(work / 'main', inp, t['tl'], t['ml'])
            if r['status'] == 'TLE':  # host-load guard: a TLE counts only if it repeats on an immediate re-run
                sb.cleanup(r)
                r = sb.run(work / 'main', inp, t['tl'], t['ml'])
            max_cpu = max(max_cpu, r['cpu_s'])
            if r['status'] != 'OK':
                sb.cleanup(r)
                return dict(base, passed=False, verdict=r['status'], gate=gate, failed_test=inp.name)
            good = sb.check(t['checker'], inp, inp.with_suffix('.out'), r['stdout_path'])
            sb.cleanup(r)
            if not good:
                return dict(base, passed=False, verdict='WA', gate=gate, failed_test=inp.name)
        return dict(base, passed=True, verdict='AC', gate=gate, max_cpu_s=max_cpu, n_tests=len(t['tests']))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('code_jsonl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--max-load', type=float, default=32.0, help='refuse to start above this 1-min load average')
    a = ap.parse_args()
    load1 = os.getloadavg()[0]
    if load1 > a.max_load:
        sys.exit(f'host 1-min load {load1:.0f} > {a.max_load:.0f}: grade on a quiet machine (CPU-time limits)')
    rows = [json.loads(x) for x in open(a.code_jsonl)]
    hy4bench.set_profile(rows[0].get('profile', 'hy4'))
    st, tasks = verify_freeze()
    want = {(t['task_id'], s) for t in st['tasks'] for s in hy4bench.CODE_SEEDS}
    got = {(r['id'], r['seed']) for r in rows}
    if got != want:
        print(f'WARNING: rows cover {len(got & want)} of {len(want)} (task, seed) pairs; missing count as fails',
              file=sys.stderr)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        graded = list(ex.map(lambda r: grade_one(r, tasks), rows))
    for tid, s in sorted(want - got):
        graded.append(dict(id=tid, tier=next(t['tier'] for t in st['tasks'] if t['task_id'] == tid), seed=s,
                           passed=False, verdict='missing', gate=False))
    with open(a.out, 'w') as f:
        for g in sorted(graded, key=lambda g: (g['id'], g['seed'])):
            f.write(json.dumps(g, sort_keys=True) + '\n')
    from collections import Counter
    v = Counter(g['verdict'] for g in graded)
    print(json.dumps(dict(load_at_start=load1, n=len(graded), passed=sum(g['passed'] for g in graded), verdicts=v,
                          gate_failures=sum(not g['gate'] for g in graded)), indent=1))


if __name__ == '__main__':
    main()
