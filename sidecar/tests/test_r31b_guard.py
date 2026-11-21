# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""R3.1b guard tests (CPU only; no vLLM needed except the optional source-contract test).

  * failfast wrapper: unit behaviour of the WorkerProc._execute_worker_rpc replacement;
  * mechanism: two real processes (torch.distributed gloo) run an RPC stream shaped like vLLM's
    (execute_model = 4 collectives, sample_tokens = 1 logits gather + 3 drafter collectives). Rank 1 raises inside
    sample_tokens once. With vLLM's stock handling the ranks' collectives pair up out of order (detected by tagging
    every collective with its op id); with the guard, rank 1 exits 86 instead;
  * acceptance-gated K controller and injection parsing;
  * source contract (VLLM_SRC=<checkout at af7f9488>): the replaced function still has the shape we re-implement.
"""
import os
import sys
from functools import partial

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hook"))
import r31b_guard as g  # noqa: E402


# ------------------------------------------------------------------------------------------- failfast unit
class _Worker:
    def __init__(self, fail_on=None):
        self.fail_on = fail_on

    def __getattr__(self, name):
        def f(*a, **k):
            if name == self.fail_on:
                raise RuntimeError(f"boom in {name}")
            return f"ok:{name}"
        return f


class _Proc:
    def __init__(self, rank, worker):
        self.rank, self.worker, self.out = rank, worker, []

    def handle_output(self, o):
        self.out.append(o)


def _stock(self, rpc_request):  # af7f9488 WorkerProc._execute_worker_rpc, verbatim logic
    method, args, kwargs, output_rank = rpc_request
    try:
        func = getattr(self.worker, method)
        output = func(*args, **kwargs)
        if output_rank is None or self.rank == output_rank:
            self.handle_output(output)
    except Exception as e:
        if output_rank is None or self.rank == output_rank:
            self.handle_output(e)


class _Exit(Exception):
    pass


@pytest.fixture
def no_exit(monkeypatch):
    def fake_exit(code):
        raise _Exit(code)
    monkeypatch.setattr(g.os, "_exit", fake_exit)
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    monkeypatch.setattr(g, "FAILFAST", True)


def _wrapped():
    return g.make_execute_worker_rpc(_stock, lambda b: None, partial)


def test_success_only_output_rank_replies(no_exit):
    rpc = _wrapped()
    p0, p1 = _Proc(0, _Worker()), _Proc(1, _Worker())
    rpc(p0, ("sample_tokens", (), {}, 0))
    rpc(p1, ("sample_tokens", (), {}, 0))
    assert p0.out == ["ok:sample_tokens"] and p1.out == []


def test_stock_swallows_on_non_output_rank():
    p1 = _Proc(1, _Worker(fail_on="sample_tokens"))
    _stock(p1, ("sample_tokens", (), {}, 0))  # returns normally: the hazard
    assert p1.out == []


def test_failfast_non_output_rank_exits(no_exit):
    rpc = _wrapped()
    p1 = _Proc(1, _Worker(fail_on="sample_tokens"))
    with pytest.raises(_Exit) as ei:
        rpc(p1, ("sample_tokens", (), {}, 0))
    assert ei.value.args[0] == g.FAILFAST_CODE and p1.out == []


def test_failfast_output_rank_reports_then_exits(no_exit):
    rpc = _wrapped()
    p0 = _Proc(0, _Worker(fail_on="execute_model"))
    with pytest.raises(_Exit):
        rpc(p0, ("execute_model", (), {}, 0))
    assert len(p0.out) == 1 and isinstance(p0.out[0], RuntimeError)


def test_non_collective_methods_keep_stock_behaviour(no_exit):
    rpc = _wrapped()
    p1 = _Proc(1, _Worker(fail_on="get_kv_cache_spec"))
    rpc(p1, ("get_kv_cache_spec", (), {}, None))
    assert len(p1.out) == 1 and isinstance(p1.out[0], RuntimeError)


def test_failfast_off_is_stock(monkeypatch, no_exit):
    monkeypatch.setattr(g, "FAILFAST", False)
    rpc = _wrapped()
    p1 = _Proc(1, _Worker(fail_on="sample_tokens"))
    rpc(p1, ("sample_tokens", (), {}, 0))
    assert p1.out == []


# --------------------------------------------------------------------------------- 2-process mechanism
STEPS = 6
FAIL_STEP = 3


def _mech_worker(rank, port, use_guard, q):
    import torch
    import torch.distributed as dist
    sys.path.insert(0, os.path.join(ROOT, "hook"))
    import r31b_guard as gg
    gg.FAILFAST = True
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
                            timeout=__import__("datetime").timedelta(seconds=20))
    log = []

    def coll(op_id):
        t = torch.tensor([float(op_id)])
        dist.all_reduce(t)
        log.append((op_id, t.item()))

    class W:
        step = 0

        def execute_model(self):
            W.step += 1
            for i in range(4):
                coll(1000 * W.step + i)

        def sample_tokens(self):
            coll(1000 * W.step + 10)               # logits gather
            if rank == 1 and W.step == FAIL_STEP:
                raise RuntimeError("drafter JIT load failed (simulated CUDA_ERROR_OUT_OF_MEMORY)")
            for i in range(3):                      # drafter forward
                coll(1000 * W.step + 20 + i)

    proc = _Proc(rank, W())
    rpc = gg.make_execute_worker_rpc(_stock, lambda b: None, partial) if use_guard else _stock
    try:
        for _ in range(STEPS):
            rpc(proc, ("execute_model", (), {}, 0))
            rpc(proc, ("sample_tokens", (), {}, 0))
    except Exception as e:  # gloo may detect the mismatch late; record it
        log.append(("error", repr(e)[:80]))
    q.put((rank, log))
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _run_mech(use_guard):
    import multiprocessing as mp
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_mech_worker, args=(r, port, use_guard, q)) for r in (0, 1)]
    for p in ps:
        p.start()
    logs = {}
    for p in ps:
        p.join(60)
    while not q.empty():
        r, log = q.get()
        logs[r] = log
    codes = [p.exitcode for p in ps]
    for p in ps:
        if p.is_alive():
            p.kill()
    return codes, logs


@pytest.mark.skipif(os.environ.get("R31B_MECH", "1") != "1", reason="R31B_MECH=0")
def test_mechanism_stock_desyncs():
    pytest.importorskip("torch")
    codes, logs = _run_mech(use_guard=False)
    # every collective on rank 0 must have seen the SAME op id on rank 1: sum == 2 * id. Stock pairs them out of order.
    mism = [(op, v) for op, v in logs.get(0, []) if op != "error" and v != 2 * op]
    assert mism, f"expected out-of-order pairing, got none; logs={logs}"
    first = mism[0]
    assert first[0] == 1000 * FAIL_STEP + 20, first  # rank 0's first drafter collective paired with rank 1's step N+1


@pytest.mark.skipif(os.environ.get("R31B_MECH", "1") != "1", reason="R31B_MECH=0")
def test_mechanism_guard_failfast():
    pytest.importorskip("torch")
    codes, logs = _run_mech(use_guard=True)
    assert codes[1] == g.FAILFAST_CODE, codes
    mism = [(op, v) for op, v in logs.get(0, []) if op != "error" and v != 2 * op]
    assert not mism, mism  # rank 0 never completes a mispaired collective; it errors/timeouts instead


# ------------------------------------------------------------------------------------- K controller etc.
def test_parse_inject():
    assert g.parse_inject("") is None
    assert g.parse_inject("rank=1,after=7") == {"rank": 1, "after": 7}


def _lookup(k_by_bs):
    return [0] + k_by_bs


def test_kctl_stays_on_with_good_acceptance():
    c = g.AcceptanceGatedK(_lookup([5] * 4 + [3] * 12 + [0]), k_off=0)
    for _ in range(50):
        for _ in range(4):
            c.observe(5, 2)                       # acceptance length 3.0
        assert c[1] == 5 and c[16] == 3
    assert c.on and c.flips == 0


def test_kctl_turns_off_probes_and_recovers():
    c = g.AcceptanceGatedK(_lookup([5] * 16 + [0]), k_off=0, probe_every=8)
    ks = []
    for _ in range(40):                            # random ids: acceptance length ~1.02
        for _ in range(4):
            c.observe(5, 0)
        ks.append(c[1])
    assert not c.on and ks.count(0) > 25 and ks.count(5) >= 4  # mostly K_OFF, periodic K=5 probes
    probes = [i for i, k in enumerate(ks) if k == 5]
    for _ in range(30):                            # traffic turns real: probes see acceptance 3.0
        for _ in range(4):
            c.observe(5, 2)
        c[1]
    assert c.on and c[1] == 5 and c.flips >= 2
    assert probes  # sanity


def test_kctl_off_choice_never_leaves_schedule():
    c = g.AcceptanceGatedK(_lookup([5] * 16 + [0]), k_off=0)
    seen = set()
    for i in range(200):
        for _ in range(4):
            c.observe(5, 0 if i < 100 else 3)
        seen.add(c[1 + i % 16])
    assert seen <= {0, 5}


# ------------------------------------------------------------------------------------- source contract
@pytest.mark.skipif(not os.environ.get("VLLM_SRC"), reason="set VLLM_SRC to a vLLM checkout at af7f9488")
def test_source_contract_af7f9488():
    import ast
    src = os.environ["VLLM_SRC"]
    mp_src = open(os.path.join(src, "vllm/v1/executor/multiproc_executor.py")).read()
    tree = ast.parse(mp_src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_execute_worker_rpc")
    body = ast.get_source_segment(mp_src, fn)
    for needle in ("method, args, kwargs, output_rank = rpc_request", "cloudpickle.loads(method)",
                   "output_rank is None or self.rank == output_rank", "self.handle_output(output)",
                   "self.handle_output(e)"):
        assert needle in body, needle
    assert "import cloudpickle" in mp_src and "from functools import partial" in mp_src
    gw = open(os.path.join(src, "vllm/v1/worker/gpu_worker.py")).read()
    assert "from vllm.model_executor.warmup.kernel_warmup import kernel_warmup" in gw and "kernel_warmup(self)" in gw
    kw = open(os.path.join(src, "vllm/model_executor/warmup/kernel_warmup.py")).read()
    assert "deep_gemm_warmup(model, max_tokens)" in kw and "model = worker.get_model()" in kw  # target only
    dw = open(os.path.join(src, "vllm/model_executor/warmup/deep_gemm_warmup.py")).read()
    for needle in ("def _deep_gemm_linear_data(", "def _generate_optimal_warmup_m_values(",
                   "get_mk_alignment_for_contiguous_layout,", "fp8_gemm_nt,"):
        assert needle in dw, needle
    df = open(os.path.join(src, "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py")).read()
    assert "class DFlashSpeculator(" in df and "def propose(" in df
    assert "self._precompute_context_kv(num_context, num_target_tokens)" in df  # the eager trigger path
    sc = open(os.path.join(src, "vllm/v1/core/sched/scheduler.py")).read()
    assert "num_spec_tokens_to_schedule = self.dynamic_sd_lookup[" in sc and "def make_spec_decoding_stats(" in sc
    cu = open(os.path.join(src, "vllm/v1/worker/gpu/cudagraph_utils.py")).read()
    assert "from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup" in cu
    assert cu.count("dense_schedule") == 2 and "for num_spec in dense_schedule[1:]" in cu  # used only as a K set
    st = open(os.path.join(src, "vllm/v1/spec_decode/dynamic/utils.py")).read()
    assert "values must be >= 0" in st  # K=0 is a legal schedule entry


def test_cudagraph_patch_adds_k_off(monkeypatch):
    import types
    monkeypatch.setattr(g, "KCTL", True)
    mod = types.SimpleNamespace(build_dynamic_sd_schedule_lookup=lambda *a, **k: [0, 5, 5, 3, 3])
    g._patch_cudagraph_utils(mod)
    assert mod.build_dynamic_sd_schedule_lookup() == [0, 5, 5, 3, 3, 1]  # K_OFF 0 -> 1
    mod2 = types.SimpleNamespace(build_dynamic_sd_schedule_lookup=lambda *a, **k: [0, 5, 1])
    g._patch_cudagraph_utils(mod2)
    assert mod2.build_dynamic_sd_schedule_lookup() == [0, 5, 1]
