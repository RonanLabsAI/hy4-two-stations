"""L6-4g Part 3: compile and run C++ in a bwrap sandbox (no network, read-only system, AtCoder limits).

Used both to build/validate the frozen tests and to grade model programs, so the two use one code path.
"""
import os, re, resource, shutil, subprocess, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / '.runner'


def _runner():
    src = HERE / 'runner.c'
    if not RUNNER.exists() or RUNNER.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(['gcc', '-O2', '-static', str(src), '-o', str(RUNNER)], check=True)
    return RUNNER

GXX = ['g++', '-std=gnu++20', '-O2', '-pipe']
BWRAP = ['bwrap', '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib', '--ro-bind', '/lib64', '/lib64',
         '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp', '--unshare-all', '--die-with-parent',
         '--new-session']


def compile_cpp(src, out, timeout=120):
    """Compile src (path) to out. Returns (ok, stderr_tail)."""
    try:
        r = subprocess.run(GXX + [str(src), '-o', str(out)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, 'compile timeout'
    return r.returncode == 0, r.stderr[-2000:]


def run(binary, stdin_path, tl_s, ml_mib, out_limit=256 << 20, wall_factor=3.0):
    """Run binary in the sandbox. Returns dict(status, cpu_s, wall_s, stdout_path, rc).

    status: OK | TLE | MLE_OR_RE | RE | OLE. CPU time (user+sys, includes descendants) is compared to tl_s.
    Memory is capped with RLIMIT_AS at ml_mib (+64 MiB slack for the loader), so an over-limit allocation
    shows as a runtime error.
    """
    binary = Path(binary).resolve()
    work = Path(tempfile.mkdtemp(prefix='p3run-'))
    shutil.copy(binary, work / 'prog')
    outp, errp = work / 'stdout', work / 'report'
    as_bytes = (ml_mib + 64) << 20
    cpu_cap = int(tl_s) + 2

    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_cap, cpu_cap))
        resource.setrlimit(resource.RLIMIT_FSIZE, (out_limit, out_limit))
        resource.setrlimit(resource.RLIMIT_STACK, (as_bytes, as_bytes))  # AtCoder stack = memory limit

    cmd = BWRAP + ['--ro-bind', str(work / 'prog'), '/prog', '--ro-bind', str(_runner()), '/runner',
                   '--chdir', '/tmp', '/runner']
    t0 = time.monotonic()
    killed = False
    with open(stdin_path, 'rb') as fi, open(outp, 'wb') as fo, open(errp, 'wb') as fe:
        p = subprocess.Popen(cmd, stdin=fi, stdout=fo, stderr=fe, preexec_fn=limits)
        deadline = t0 + tl_s * wall_factor + 2
        while True:
            pid, st, ru = os.wait4(p.pid, os.WNOHANG)
            if pid:
                break
            if time.monotonic() > deadline:
                killed = True
                p.kill()
                pid, st, ru = os.wait4(p.pid, 0)
                break
            time.sleep(0.005)
    wall = time.monotonic() - t0
    m = re.search(r'P3RUN cpu=([\d.]+) rss_kib=(\d+) exit=(-?\d+) signal=(\d+)', errp.read_text(errors='replace'))
    if m:
        cpu, rss_kib, rc, sig = float(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    else:  # runner itself killed (wall deadline)
        cpu, rss_kib, rc, sig = float('inf') if killed else 0.0, 0, -1, 9
    if killed or cpu > tl_s or sig in (9, 24):  # 24 = SIGXCPU
        status = 'TLE'
    elif outp.stat().st_size >= out_limit:
        status = 'OLE'
    elif rss_kib > ml_mib * 1024:
        status = 'MLE'
    elif rc != 0 or sig:
        status = 'RE'
    else:
        status = 'OK'
    return dict(status=status, cpu_s=round(cpu, 3), wall_s=round(wall, 3), rc=rc, rss_mib=rss_kib // 1024, stdout_path=str(outp),
                workdir=str(work))


def tokens_equal(a_path, b_path):
    """AtCoder default judge: whitespace-separated tokens compared exactly."""
    return Path(a_path).read_bytes().split() == Path(b_path).read_bytes().split()


def check(checker_bin, in_path, ans_path, out_path, timeout=30):
    """Custom checker: `checker input expected contestant` exits 0 to accept. Runs unsandboxed (our code)."""
    if checker_bin is None:
        return tokens_equal(ans_path, out_path)
    try:
        r = subprocess.run([str(checker_bin), str(in_path), str(ans_path), str(out_path)],
                           capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0


def cleanup(res):
    shutil.rmtree(res['workdir'], ignore_errors=True)
