#!/usr/bin/env python3
"""L6-4g Part 3 (Amendment 2, "Set"): write data/part3/SET.json, the frozen task list.

Includes every fetched task whose BUILD.json is ok and which is not interactive; lists the excluded ones with
reasons. Records each task's statement hash, BUILD.json hash, limits and tier, plus the toolchain versions.
Run after every build_tests.py run is final, and before build_inputs.py renders code_atcoder.json.
"""
import hashlib, json, platform, subprocess
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data' / 'part3'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tier(tid):
    c, letter = tid.split('_')
    if c.startswith('abc'):
        return 'abc-ad' if letter in 'abcd' else 'abc-eg'
    return 'arc-agc'


def main():
    tasks, excluded = [], []
    for d in sorted(p for p in (DATA / 'tasks').iterdir() if p.is_dir()):
        m = json.loads((d / 'meta.json').read_text())
        b = json.loads((d / 'BUILD.json').read_text()) if (d / 'BUILD.json').exists() else None
        if m['interactive']:
            excluded.append(dict(task_id=d.name, reason='interactive'))
        elif b is None:
            excluded.append(dict(task_id=d.name, reason='no BUILD.json'))
        elif b['status'] != 'ok':
            excluded.append(dict(task_id=d.name, reason='test validation: ' + b['reason']))
        else:
            if sha(d / 'statement.txt') != m['statement_sha256']:
                raise SystemExit(f'{d.name}: statement changed since fetch')
            tasks.append(dict(task_id=d.name, contest=m['contest'], tier=tier(d.name), title=m['title'],
                              time_limit_s=m['time_limit_s'], memory_limit_mib=m['memory_limit_mib'],
                              statement_sha256=m['statement_sha256'], build_sha256=sha(d / 'BUILD.json'),
                              n_tests=len(b['tests'])))
    gxx = subprocess.run(['g++', '--version'], capture_output=True, text=True).stdout.splitlines()[0]
    out = dict(rule='rated ABC/ARC/AGC contests starting after 2026-11-04 23:59 PT, ended >= 72 h before freeze; '
                    'AHC and interactive tasks excluded (Amendment 2)',
               contests=sorted({t['contest'] for t in tasks} | {e['task_id'].split('_')[0] for e in excluded}),
               tasks=tasks, excluded=excluded, toolchain=dict(gxx=gxx, python=platform.python_version()))
    (DATA / 'SET.json').write_text(json.dumps(out, indent=1) + '\n')
    print(f"{len(tasks)} tasks, {len(excluded)} excluded: {excluded}")


if __name__ == '__main__':
    main()
