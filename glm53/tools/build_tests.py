#!/usr/bin/env python3
"""L6-4g Part 3 (Amendment 2, "Tests"): build and validate one task's frozen test set.

Task dir tools/hy4/data/part3/tasks/<task_id>/ must hold (written per task before any model output exists):
  ref.cpp       reference from the official editorial (transcribed code, or written from its explanation)
  brute.cpp     independent brute force, correct on small inputs (optional only if checker.cpp fully verifies
                an answer without needing the optimum; BUILD.json then records brute = "checker-verified")
  gen.py        `python3 gen.py tests OUTDIR` writes >= 30 inputs OUTDIR/gNN.in (deterministic, fixed seeds;
                small random, max-size random, constructed edge cases);
                `python3 gen.py small SEED` prints one small random input to stdout
  validate.py   reads one input on stdin, exits 0 iff it satisfies every stated constraint and the format
  checker.cpp   optional; `checker input expected contestant` exits 0 to accept (multi-answer or float tasks).
                `expected` is the reference's (or brute's) output.

Steps (all must pass, else status = "dropped" with the reason):
  1. compile; 2. samples: validator accepts, reference matches official sample output;
  3. generated tests: >= 30, validator accepts each, reference output written as expected, reference CPU <= TL/2;
  4. >= 200 small random inputs: validator accepts, reference agrees with brute (via checker if present).
Writes tests/ (sNN = samples, gNN = generated) and BUILD.json with SHA-256 of every source and test file.
Usage: build_tests.py TASK_ID [TASK_ID ...]   (or --all)
"""
import concurrent.futures as cf, hashlib, json, shutil, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sandbox as sb

TASKS = Path(__file__).resolve().parent.parent / 'data' / 'part3' / 'tasks'
N_GEN_MIN, N_SMALL = 30, 200


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(d, inp):
    with open(inp, 'rb') as f:
        r = subprocess.run([sys.executable, str(d / 'validate.py')], stdin=f, capture_output=True, timeout=120)
    return r.returncode == 0, r.stderr.decode(errors='replace')[-300:]


def build(tid):
    d = TASKS / tid
    meta = json.loads((d / 'meta.json').read_text())
    tl, ml = meta['time_limit_s'], meta['memory_limit_mib']
    rep = dict(task_id=tid, status='ok', reason=None)
    bins = Path(tempfile.mkdtemp(prefix=f'p3b-{tid}-'))

    def drop(reason):
        rep.update(status='dropped', reason=reason)
        return rep

    for f in ('ref.cpp', 'gen.py', 'validate.py'):
        if not (d / f).exists():
            return drop(f'missing {f}')
    ok, err = sb.compile_cpp(d / 'ref.cpp', bins / 'ref')
    if not ok:
        return drop('ref compile: ' + err[-300:])
    have_brute = (d / 'brute.cpp').exists()
    have_checker = (d / 'checker.cpp').exists()
    if have_brute:
        ok, err = sb.compile_cpp(d / 'brute.cpp', bins / 'brute')
        if not ok:
            return drop('brute compile: ' + err[-300:])
    elif not have_checker:
        return drop('no brute.cpp and no checker.cpp')
    chk = None
    if have_checker:
        ok, err = sb.compile_cpp(d / 'checker.cpp', bins / 'checker')
        if not ok:
            return drop('checker compile: ' + err[-300:])
        chk = bins / 'checker'

    tests = d / 'tests'
    if tests.exists():
        shutil.rmtree(tests)
    tests.mkdir()
    # 2. samples
    for s_in in sorted((d / 'samples').glob('*.in')):
        n = s_in.stem
        shutil.copy(s_in, tests / f's{n}.in')
        shutil.copy(s_in.with_suffix('.out'), tests / f's{n}.out')
        ok, why = validate(d, s_in)
        if not ok:
            return drop(f'validator rejects sample {n}: {why}')
        r = sb.run(bins / 'ref', s_in, tl, ml)
        good = r['status'] == 'OK' and sb.check(chk, s_in, s_in.with_suffix('.out'), r['stdout_path'])
        sb.cleanup(r)
        if not good:
            return drop(f'reference fails sample {n} ({r["status"]})')
    # 3. generated tests
    g = subprocess.run([sys.executable, str(d / 'gen.py'), 'tests', str(tests)], capture_output=True, timeout=1800)
    if g.returncode != 0:
        return drop('gen tests: ' + g.stderr.decode(errors='replace')[-300:])
    gen = sorted(tests.glob('g*.in'))
    if len(gen) < N_GEN_MIN:
        return drop(f'only {len(gen)} generated tests')

    def one(inp):
        ok, why = validate(d, inp)
        if not ok:
            return inp.name, f'validator rejects: {why}', 0.0
        r = sb.run(bins / 'ref', inp, tl, ml, wall_factor=4.0)
        shutil.copy(r['stdout_path'], inp.with_suffix('.out'))
        st, cpu = r['status'], r['cpu_s']
        sb.cleanup(r)
        if st != 'OK':
            return inp.name, f'reference {st}', cpu
        if cpu > tl / 2:
            return inp.name, f'reference cpu {cpu}s > TL/2', cpu
        if chk is not None and not sb.check(chk, inp, inp.with_suffix('.out'), inp.with_suffix('.out')):
            return inp.name, 'checker rejects reference output', cpu
        return inp.name, None, cpu

    with cf.ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, gen))
    bad = [(n, e) for n, e, _ in res if e]
    if bad:
        return drop(f'generated test {bad[0][0]}: {bad[0][1]} ({len(bad)} bad)')
    rep['ref_max_cpu_s'] = max(c for _, _, c in res)

    # 4. small cross-check
    def small(seed):
        work = Path(tempfile.mkdtemp(prefix='p3s-'))
        try:
            g = subprocess.run([sys.executable, str(d / 'gen.py'), 'small', str(seed)], capture_output=True,
                               timeout=60)
            if g.returncode != 0:
                return seed, 'gen small failed'
            inp = work / 'in'
            inp.write_bytes(g.stdout)
            ok, why = validate(d, inp)
            if not ok:
                return seed, f'validator rejects small: {why}'
            rr = sb.run(bins / 'ref', inp, max(tl, 5), ml)
            if rr['status'] != 'OK':
                return seed, f'ref {rr["status"]}'
            if have_brute:
                rb = sb.run(bins / 'brute', inp, 20, ml)
                if rb['status'] != 'OK':
                    return seed, f'brute {rb["status"]}'
                agree = sb.check(chk, inp, rb['stdout_path'], rr['stdout_path'])
                sb.cleanup(rb)
            else:
                agree = sb.check(chk, inp, rr['stdout_path'], rr['stdout_path'])
            sb.cleanup(rr)
            return seed, None if agree else 'ref and brute disagree'
        finally:
            shutil.rmtree(work, ignore_errors=True)

    with cf.ThreadPoolExecutor(16) as ex:
        sres = list(ex.map(small, range(1, N_SMALL + 1)))
    sbad = [(s, e) for s, e in sres if e]
    if sbad:
        return drop(f'small seed {sbad[0][0]}: {sbad[0][1]} ({len(sbad)} of {N_SMALL} bad)')

    rep.update(n_samples=len(list(tests.glob('s*.in'))), n_generated=len(gen), n_small_crosscheck=N_SMALL,
               brute='brute.cpp' if have_brute else 'checker-verified', checker=have_checker,
               time_limit_s=tl, memory_limit_mib=ml,
               sources={f: sha(d / f) for f in ('ref.cpp', 'brute.cpp', 'gen.py', 'validate.py', 'checker.cpp')
                        if (d / f).exists()},
               tests={p.name: sha(p) for p in sorted(tests.iterdir())})
    shutil.rmtree(bins, ignore_errors=True)
    return rep


def main():
    ids = sorted(p.name for p in TASKS.iterdir() if p.is_dir()) if sys.argv[1:] == ['--all'] else sys.argv[1:]
    for tid in ids:
        rep = build(tid)
        (TASKS / tid / 'BUILD.json').write_text(json.dumps(rep, indent=1) + '\n')
        extra = f"gen={rep.get('n_generated')} ref_max_cpu={rep.get('ref_max_cpu_s')}" if rep['status'] == 'ok' \
            else rep['reason']
        print(f"{tid}\t{rep['status']}\t{extra}", flush=True)


if __name__ == '__main__':
    main()
