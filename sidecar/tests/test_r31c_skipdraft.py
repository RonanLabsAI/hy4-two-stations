# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""R3.1c skip-draft tests (CPU only; vLLM needed only for the optional source-contract test).

  * controller: K_OFF=0, probe back-off, per-probe folding, hysteresis, recovery;
  * cudagraph patch: K=0 decode graphs for the target, none for the DSpark drafter (whose K=0 query length is 0);
  * worker wrappers: execute_model records the broadcast K, propose() takes the light path only at K=0 on a real,
    async-scheduled step, and the light path runs the real light_propose() (staging, prepare per group, eager
    context precompute of every target row, no draft forward);
  * rank agreement: a scheduler process drives AcceptanceGatedK over mixed traffic and broadcasts SchedulerOutput-like
    objects to two worker processes (torch.distributed gloo). Each worker runs the REAL guard wrappers around a fake
    runner/speculator whose target forward, logits gather and draft forward are op-id-tagged collectives. With the
    scheduler-side decision every collective pairs with its twin and both ranks take identical light/full sequences;
    the negative control (each rank decides from its own state, as in vllm#54011) mispairs;
  * source contract (VLLM_SRC=<checkout at af7f9488>): every vLLM fact the light path and the K semantics rely on.
"""
import os
import random
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hook"))
import r31b_guard as g  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_skip():
    saved = dict(g._skip)
    yield
    g._skip.clear()
    g._skip.update(saved)


# ------------------------------------------------------------------------------------------- controller
def _lookup(k_by_bs):
    return [0] + k_by_bs


def _drive(c, steps, bs, accept, rng, ks=None):
    """One schedule() per step; the drafts scheduled at step i are verified (observed) at step i+1, as in vLLM."""
    pending = 0
    ks = [] if ks is None else ks
    for _ in range(steps):
        if pending:
            for _ in range(bs):
                c.observe(pending, accept(pending, rng))
        k = c[bs]
        ks.append(k)
        pending = k
    return ks


def _rand_ids(k, rng):  # catid-like: acceptance length ~1.02
    return 1 if rng.random() < 0.02 else 0


def _real(k, rng):      # real text: acceptance length ~2.9
    return min(k, 2 if rng.random() < 0.9 else 1)


def test_k_off_zero_controller_turns_off_and_backs_off():
    rng = random.Random(1)
    c = g.AcceptanceGatedK(_lookup([5] * 16), k_off=0, probe_every=16, probe_max=256)
    ks = _drive(c, 3000, 1, _rand_ids, rng)
    assert not c.on and set(ks) <= {0, 5}
    starts = [i for i in range(51, len(ks)) if ks[i] == 5 and ks[i - 1] == 0]
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert gaps == sorted(gaps) and gaps[-1] == 256 + 3, gaps  # 16, 32, ... capped at 256 (+ the 4-step burst)
    assert all(ks[s:s + 4] == [5] * 4 and ks[s + 4] == 0 for s in starts[:-1])  # C1 burst = min_obs steps
    assert sum(1 for k in ks[500:] if k) / len(ks[500:]) < 0.02  # 4 probe steps per 260 (~1.5%) once backed off


def test_k_off_zero_recovers_on_first_probe_after_traffic_turns_real():
    rng = random.Random(2)
    c = g.AcceptanceGatedK(_lookup([5] * 16), k_off=0, probe_every=16, probe_max=256)
    _drive(c, 3000, 1, _rand_ids, rng)
    assert not c.on and c.probe_gap == 256
    ks = _drive(c, 300, 1, _real, rng)
    on_at = next(i for i in range(len(ks)) if all(k == 5 for k in ks[i:i + 20]))
    assert c.on and on_at <= 256 + 8, on_at  # the first probe burst (mean ~2.9 > 1.9) turns it back on
    assert c.probe_gap == 256  # reset happens on the next ON->OFF transition


def test_hysteresis_band_does_not_flap():
    rng = random.Random(3)
    mid = lambda k, r: 1 if r.random() < 0.7 else 0  # noqa: E731  acceptance length ~1.7: inside (1.5, 1.9)
    c = g.AcceptanceGatedK(_lookup([5] * 16), k_off=0, probe_max=256)
    _drive(c, 2000, 1, mid, rng)
    assert c.on and c.flips == 0                   # ON stays ON inside the band
    _drive(c, 300, 1, _rand_ids, rng)
    assert not c.on
    f = c.flips
    _drive(c, 3000, 1, mid, rng)
    assert not c.on and c.flips == f               # OFF stays OFF inside the band


def test_c16_turns_off_and_recovers():
    rng = random.Random(4)
    c = g.AcceptanceGatedK(_lookup([5] * 4 + [3] * 12), k_off=0, probe_max=256)
    ks = _drive(c, 400, 16, _rand_ids, rng)
    assert not c.on and set(ks) <= {0, 3}
    ks = _drive(c, 300, 16, _real, rng)
    assert c.on and ks[-1] == 3


def test_r31b_semantics_kept_without_probe_max():
    c = g.AcceptanceGatedK(_lookup([5] * 16), k_off=1, probe_every=8)
    assert c.probe_max == 8
    ks = []
    for _ in range(60):
        for _ in range(4):
            c.observe(5, 0)
        ks.append(c[1])
    assert set(ks) <= {1, 5} and ks.count(5) >= 5


# ------------------------------------------------------------------------------------------ cudagraph patch
class _Mgr:
    """Shape of a CudaGraphManager calling build_dynamic_sd_schedule_lookup from _init_candidates."""

    def __init__(self, mod, decode_query_len, num_spec):
        self.mod, self.decode_query_len = mod, decode_query_len
        self.vllm_config = types.SimpleNamespace(num_speculative_tokens=num_spec)

    def _init_candidates(self):
        dense_schedule = self.mod.build_dynamic_sd_schedule_lookup([[1, 4, 5]])
        per_step = self.decode_query_len - self.vllm_config.num_speculative_tokens
        return sorted({k + per_step for k in dense_schedule[1:]})


def test_cudagraph_patch_k0_target_only(monkeypatch):
    monkeypatch.setattr(g, "KCTL", True)
    monkeypatch.setattr(g, "SKIPDRAFT", True)
    monkeypatch.setenv("R31B_KCTL_KOFF", "0")
    mod = types.SimpleNamespace(build_dynamic_sd_schedule_lookup=lambda *a, **k: [0, 5, 5, 3, 3])
    g._patch_cudagraph_utils(mod)
    assert _Mgr(mod, 6, 5)._init_candidates() == [1, 4, 6]   # target: K=0 -> plain one-token decode graphs
    assert _Mgr(mod, 5, 5)._init_candidates() == [3, 5]      # DSpark drafter: no query length 0
    assert mod.build_dynamic_sd_schedule_lookup() == [0, 5, 5, 3, 3, 0]  # unknown caller: append (target-safe)


def test_k_off_zero_still_clamped_without_skipdraft(monkeypatch):
    monkeypatch.setattr(g, "SKIPDRAFT", False)
    monkeypatch.setenv("R31B_KCTL_KOFF", "0")
    assert g.kctl_k_off() == 1
    monkeypatch.setattr(g, "SKIPDRAFT", True)
    monkeypatch.setenv("R31B_KCTL_KOFF", "0")
    assert g.kctl_k_off() == 0


# ------------------------------------------------------------------------------------------ worker wrappers
def _fake_spec(torch, n_groups=2, hidden=8, max_tokens=64, max_reqs=8, calls=None):
    calls = [] if calls is None else calls
    spec = types.SimpleNamespace()
    spec.vllm_config = types.SimpleNamespace(scheduler_config=types.SimpleNamespace(async_scheduling=True))
    spec.model = types.SimpleNamespace(combine_hidden_states=lambda x: (calls.append(("combine", x.shape[0])),
                                                                          x[:, :hidden])[1])
    spec.hidden_states = torch.zeros(max_tokens, hidden)
    spec.prepare_context_anchor = lambda ib, nr: calls.append(("anchor",))
    spec.pcp_manager = None
    spec.draft_kv_cache_group_ids = list(range(n_groups))
    spec.block_tables = types.SimpleNamespace(
        slot_mappings=[f"slots{i}" for i in range(n_groups)], input_block_tables=[f"bt{i}" for i in range(n_groups)],
        kernel_block_sizes=[256] * n_groups, cp_rank=0, cp_interleave=1)
    spec._context_slot_mappings = [f"ctx{i}" for i in range(n_groups)]
    for name in ("input_buffers", "context_positions", "sample_indices", "sample_pos", "sample_idx_mapping",
                 "temperature", "seeds"):
        setattr(spec, name, name)
    spec.dcp_size, spec.parallel_drafting_token_id, spec.num_query_per_req = 1, 7, 5
    spec.num_speculative_steps, spec.max_num_reqs, spec.max_num_tokens, spec.max_model_len = 5, max_reqs, max_tokens, 99
    spec.sample_from_anchor = True
    spec._precompute_context_kv = lambda s, e: calls.append(("precompute", s, e))
    spec.draft_tokens = torch.full((max_reqs, 5), -1)
    return spec, calls


def _propose_args(torch, num_reqs=3, num_tokens=3, hidden=8, aux=True):
    ib = types.SimpleNamespace(num_reqs=num_reqs, num_tokens=num_tokens, idx_mapping="idx")
    h = torch.randn(num_tokens, hidden)
    return (ib, "attn", "slots", torch.randn(num_tokens, hidden), [h, h] if aux else None,
            "num_sampled", "num_rejected", "last_sampled", "next_prefill", "temp_in", "seeds_in")


def test_light_propose_matches_propose_minus_draft():
    torch = pytest.importorskip("torch")
    spec, calls = _fake_spec(torch)
    prep = []
    args = _propose_args(torch, num_reqs=3, num_tokens=40)  # a mixed step: 40 target rows > graph context rows
    out = g.light_propose(spec, lambda *a: prep.append(a), dict(zip(g.PROPOSE_PARAMS, args)))
    assert [c[0] for c in calls] == ["combine", "anchor", "precompute"]
    assert calls[-1] == ("precompute", 0, 40)               # every target row, eagerly
    assert torch.equal(spec.hidden_states[:40], args[4][0])  # combined aux states staged (fake combine = slice)
    assert len(prep) == 2 and [p[1] for p in prep] == ["slots0", "slots1"] and [p[3] for p in prep] == ["ctx0", "ctx1"]
    p = prep[0]
    assert p[9] is args[0] and p[10:16] == ("num_sampled", "num_rejected", "last_sampled", "next_prefill", "temp_in",
                                            "seeds_in")
    assert p[16:] == ("bt0", 256, 0, 1, 1, 7, 5, 5, 8, 64, 99, True)
    assert out.shape == (3, 5)


@pytest.mark.parametrize("next_k,dummy,profile,async_sched,light", [
    (0, False, False, True, True), (3, False, False, True, False), (5, False, False, True, False),
    (0, True, False, True, False), (0, False, True, True, False), (0, False, False, False, False),
    (None, False, False, True, False)])
def test_skip_decision(monkeypatch, next_k, dummy, profile, async_sched, light):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(g, "SKIPDRAFT", True)
    spec, calls = _fake_spec(torch)
    spec.vllm_config.scheduler_config.async_scheduling = async_sched
    g._skip.update({"next_k": next_k, "calls": 0, "light": 0, "full": 0, "checks": 0, "async": None})
    orig_calls = []
    wrapped = g.make_skip_propose(lambda self, *a, **k: orig_calls.append(1) or "drafted", lambda *a: None)
    out = wrapped(spec, *_propose_args(torch), dummy_run=dummy, is_profile=profile)
    assert (orig_calls == []) == light
    assert (out == "drafted") != light


def test_execute_model_records_broadcast_k(monkeypatch):
    monkeypatch.setattr(g, "SKIPDRAFT", True)

    class R:
        def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False, **kw):
            return g._skip["next_k"]

    mod = types.SimpleNamespace(GPUModelRunner=R)
    g._patch_model_runner(mod)
    so = types.SimpleNamespace(num_spec_tokens_to_schedule=0)
    assert R().execute_model(so) == 0
    assert R().execute_model(types.SimpleNamespace(num_spec_tokens_to_schedule=3)) == 3
    assert R().execute_model(so, None, True) is None          # positional dummy_run
    assert R().execute_model(so, dummy_run=True) is None
    assert R().execute_model(types.SimpleNamespace()) is None  # field missing -> never skip
    g._patch_model_runner(mod)
    assert getattr(R.execute_model, "_r31c", False)


# --------------------------------------------------------------------------------- 2-process rank agreement
N_STEPS = 400


def _traffic(i):
    """Mixed traffic: real text, random ids, real again, short random burst, real; batch size varies."""
    phase = (i // 80) % 5
    bs = 1 if i < 200 else 16
    return bs, (_rand_ids if phase in (1, 3) else _real)


def _agree_worker(rank, port, mode, q_in, q_out):
    import torch
    import torch.distributed as dist
    sys.path.insert(0, os.path.join(ROOT, "hook"))
    import r31b_guard as gg
    gg.SKIPDRAFT = True
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
                            timeout=__import__("datetime").timedelta(seconds=15))
    gg._tp_cpu_group = lambda: (dist.group.WORLD, rank, 2)
    log, decisions = [], []
    local_rng = random.Random(1000 + rank)

    def coll(op_id):
        t = torch.tensor([float(op_id)])
        dist.all_reduce(t)
        log.append((op_id, t.item()))

    class Spec:
        def propose(self, *a, **k):          # the draft forward: 3 TP collectives
            for j in range(3):
                coll(1000 * W.step + 20 + j)
            return self.draft_tokens[: a[0].num_reqs]

    dmod = types.SimpleNamespace(DFlashSpeculator=Spec, prepare_dflash_inputs=lambda *a: None)
    gg.CHECK_EVERY = 0 if mode == "local" else 8
    gg._patch_dflash_skip(dmod)              # the real wrapper, as installed on DFlashSpeculator
    fake = Spec()
    fake.__dict__.update(_fake_spec(torch)[0].__dict__)

    class W:
        step = 0

        def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False):
            W.step = scheduler_output.step
            for j in range(4):               # target forward: 4 TP collectives
                coll(1000 * W.step + j)

        def sample_tokens(self):
            coll(1000 * W.step + 10)         # logits gather
            if mode == "local":              # vllm#54011 shape: each rank decides from its own state
                gg._skip["next_k"] = 0 if local_rng.random() < 0.5 else 5
            before = gg._skip["light"]
            fake.propose(*_propose_args(torch, num_reqs=2, num_tokens=2))
            decisions.append("L" if gg._skip["light"] > before else "F")

    gg._patch_model_runner(types.SimpleNamespace(GPUModelRunner=W))
    w = W()
    err = None
    try:
        while True:
            so = q_in.get(timeout=60)
            if so is None:
                break
            w.execute_model(so)
            w.sample_tokens()
    except Exception as e:
        err = repr(e)[:120]
    q_out.put((rank, log, decisions, dict(gg._skip), err))
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _run_agree(mode):
    import multiprocessing as mp
    import pickle
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    ctx = mp.get_context("spawn")
    qs = [ctx.Queue() for _ in range(2)]
    q_out = ctx.Queue()
    ps = [ctx.Process(target=_agree_worker, args=(r, port, mode, qs[r], q_out)) for r in (0, 1)]
    for p in ps:
        p.start()
    # The scheduler (EngineCore): decides K, broadcasts the same pickled SchedulerOutput to both ranks.
    rng = random.Random(7)
    ctl = g.AcceptanceGatedK(_lookup([5] * 4 + [3] * 12), k_off=0, probe_every=16, probe_max=64)
    ks, pending = [], 0
    for i in range(N_STEPS):
        bs, acc = _traffic(i)
        if pending:
            for _ in range(bs):
                ctl.observe(pending, acc(pending, rng))
        k = ctl[bs]
        ks.append(k)
        pending = k
        blob = pickle.dumps(types.SimpleNamespace(step=i + 1, num_spec_tokens_to_schedule=k))
        for q in qs:
            q.put(pickle.loads(blob))
    for q in qs:
        q.put(None)
    res = {}
    for _ in ps:
        try:
            r, log, dec, st, err = q_out.get(timeout=120)
            res[r] = (log, dec, st, err)
        except Exception:
            break
    for p in ps:
        p.join(30)
        if p.is_alive():
            p.kill()
    return ks, res, [p.exitcode for p in ps]


def _mispaired(log):
    return [(op, v) for op, v in log if v != 2 * op]


@pytest.mark.skipif(os.environ.get("R31B_MECH", "1") != "1", reason="R31B_MECH=0")
def test_two_rank_agreement_scheduler_side():
    pytest.importorskip("torch")
    ks, res, codes = _run_agree("scheduler")
    assert codes == [0, 0], (codes, {r: v[3] for r, v in res.items()})
    (log0, dec0, st0, err0), (log1, dec1, st1, err1) = res[0], res[1]
    assert err0 is None and err1 is None, (err0, err1)
    assert not _mispaired(log0) and not _mispaired(log1)
    assert dec0 == dec1 == ["L" if k == 0 else "F" for k in ks]
    assert 0 < dec0.count("L") < N_STEPS and dec0.count("F") > 0
    assert st0["checks"] == st1["checks"] == N_STEPS // 8 and not st0.get("check_failed")
    assert len(log0) == len(log1) == 5 * N_STEPS + 3 * dec0.count("F")  # light steps issue no collective


@pytest.mark.skipif(os.environ.get("R31B_MECH", "1") != "1", reason="R31B_MECH=0")
def test_two_rank_rank_local_decision_desyncs():
    pytest.importorskip("torch")
    _, res, _ = _run_agree("local")
    bad = [r for r, (log, dec, st, err) in res.items() if _mispaired(log) or err]
    assert bad, "expected rank-local skip decisions to mispair collectives"


# ------------------------------------------------------------------------------------------ source contract
@pytest.mark.skipif(not os.environ.get("VLLM_SRC"), reason="set VLLM_SRC to a vLLM checkout at af7f9488")
def test_source_contract_r31c_af7f9488():
    import ast
    import inspect
    import re
    src = os.environ["VLLM_SRC"]
    rd = lambda p: open(os.path.join(src, p)).read()  # noqa: E731
    out = rd("vllm/v1/core/sched/output.py")
    assert "# Number of spec tokens to schedule for the next step.\n    num_spec_tokens_to_schedule: int = 0" in out
    asch = rd("vllm/v1/core/sched/async_scheduler.py")
    assert "self._spec_token_placeholders = [\n            -1\n        ] * scheduler_output.num_spec_tokens_to_schedule" \
        in asch and "request.spec_token_ids = self._spec_token_placeholders" in asch
    sc = rd("vllm/v1/core/sched/scheduler.py")
    assert "num_spec_tokens_to_schedule=num_spec_tokens_to_schedule," in sc
    # V2 runner: execute_model(scheduler_output, intermediate_tensors, dummy_run, ...) then sample_tokens -> propose
    mr = rd("vllm/v1/worker/gpu/model_runner.py")
    tree = ast.parse(mr)
    em = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "execute_model")
    assert [a.arg for a in em.args.args[:4]] == ["self", "scheduler_output", "intermediate_tensors", "dummy_run"]
    assert "class GPUModelRunner(" in mr and "draft_tokens = self.speculator.propose(" in mr
    # propose(): parameter order, and the prepare_dflash_inputs call light_propose replicates
    df = rd("vllm/v1/worker/gpu/spec_decode/dflash/speculator.py")
    tree = ast.parse(df)
    pr = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "propose")
    assert tuple(a.arg for a in pr.args.args[1:12]) == g.PROPOSE_PARAMS
    assert [a.arg for a in pr.args.args[12:]] == ["dp_sync", "dummy_run", "skip_attn_for_dummy_run", "mm_inputs",
                                                  "is_profile"]
    body = ast.get_source_segment(df, pr)
    call = re.search(r"prepare_dflash_inputs\((.*?)\n            \)", body, re.S).group(1)
    upstream = [x.strip() for x in call.split(",") if x.strip()]
    mine = inspect.getsource(g.light_propose)
    mcall = re.search(r"prepare_dflash_inputs\((.*?)\n            \)", mine, re.S).group(1)
    ours = [x.strip() for x in mcall.split(",") if x.strip()]
    norm = {"num_sampled": 'a["num_sampled"]', "num_rejected": 'a["num_rejected"]', "last_sampled": 'a["last_sampled"]',
            "next_prefill_tokens": 'a["next_prefill_tokens"]', "temperature": 'a["temperature"]',
            "seeds": 'a["seeds"]', "input_batch": "input_batch"}
    assert [norm.get(x, x.replace("self.", "spec.")) for x in upstream] == ours
    for needle in ("hidden_states = self.model.combine_hidden_states(\n                torch.cat(aux_hidden_states, dim=-1)",
                   "self.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])",
                   "self.prepare_context_anchor(input_batch, num_rejected)",
                   "self.block_tables.gather_block_tables(\n                input_batch.idx_mapping, num_reqs_padded=num_reqs",
                   "self._precompute_context_kv(num_context, num_target_tokens)",
                   "self.query_cudagraph_manager.run_fullgraph(batch_desc)",
                   "return self.draft_tokens[:num_reqs]"):
        assert needle in body, needle
    # DSpark inherits propose and does not override prepare_context_anchor
    ds = rd("vllm/v1/worker/gpu/spec_decode/dspark/speculator.py")
    assert "class DSparkSpeculator(DFlashSpeculator)" in ds and "def propose(" not in ds
    assert "def prepare_context_anchor(" not in ds
    # The context precompute has no collective: replicated / disable_tp linears and a local cache insert.
    m = rd("vllm/models/deepseek_v4/nvidia/dspark.py")
    pc = m[m.index("def precompute_and_store_context_kv("):m.index("def forward(")]
    assert "self.context_wkv_proj(main_x)" in pc and "_insert_context_kv(" in pc
    assert "all_reduce" not in pc and "all_gather" not in pc
    cw = m[m.index("self.context_wkv_proj = MergedColumnParallelLinear("):]
    assert "disable_tp=True" in cw[:cw.index(")\n")]
    mp = m[m.index("self.main_proj = "):]
    assert mp.startswith("self.main_proj = ReplicatedLinear(")
    # Graph manager: the drafter's manager reuses the dense K set with decode_query_len - num_spec_tokens new tokens
    cu = rd("vllm/v1/worker/gpu/cudagraph_utils.py")
    ic = cu[cu.index("def _init_candidates(self)"):]
    assert "dense_schedule = build_dynamic_sd_schedule_lookup(" in ic
    assert "self.decode_query_len - self.vllm_config.num_speculative_tokens" in ic
    dc = rd("vllm/v1/worker/gpu/spec_decode/dflash/speculator.py")
    assert "decode_query_len=self.num_query_per_req," in dc
    # Async scheduling is on by default for dspark (mp executor), which the light step requires.
    vc = rd("vllm/config/vllm.py")
    assert 'and self.speculative_config.method != "dspark"' in vc
