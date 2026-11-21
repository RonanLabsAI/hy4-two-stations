# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""R3.1b guard for multi-node DSpark on vLLM nightly af7f9488 (DeepSeek-V4-Pro, TP2+EP2 across two GB300 Stations).

Root cause of the R3.1 GSM8K hang (R31B-DSPARK-HANG.md):
  rank 1's DeepGEMM JIT failed to LOAD a never-before-used kernel (cuKernelGetFunction -> CUDA_ERROR_OUT_OF_MEMORY at
  --gpu-memory-utilization 0.99) inside the DSpark drafter's eager context-KV precompute (context_wkv_proj, M = target
  tokens - graph context rows, a value only a mixed prefill step produces). vLLM's WorkerProc._execute_worker_rpc logs
  an exception on a NON-output rank and simply dequeues the next RPC. Rank 1 therefore skipped the rest of
  sample_tokens(N) (the drafter forward and its collectives) and ran execute_model(N+1), whose all-reduces paired with
  rank 0's drafter all-reduces: a cross-node collective desync. Receipt: rank 1 published exactly 3 more E3 calls than
  rank 0, all at T=96 (= 16 reqs x 6 verify tokens; 288 tokens = 3 x 96) -> 3 target layers of step N+1.

What this module installs (each part is independent; all are rank-symmetric):
  1. FAILFAST (R31B_FAILFAST=1, default): an exception inside execute_model / sample_tokens / execute_dummy_batch on
     ANY rank of a multi-rank engine is fatal: the traceback is logged as "R31B FAILFAST" and the worker exits
     (os._exit(86)). A silent desync becomes a loud, attributable failure. (Pair it with a shorter
     VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: the peer rank can only notice through the engine's RPC timeout.)
  2. DRAFTER DEEPGEMM WARM-UP (R31B_DRAFT_DG_WARMUP=relax|full|skip, default relax): after vLLM's kernel_warmup (which
     warms only the TARGET model's DeepGEMM linears), run the same fp8_gemm_nt warm-up over every DeepGEMM linear of
     the speculator's draft model, so no drafter GEMM ever JIT-compiles/loads in the serving hot path.
     "relax" = vLLM's own M set (covers every kernel config), "full" = every M in 1..max_num_batched_tokens.
  3. MEMORY TELEMETRY (R31B_STATS=path): a daemon thread records torch.cuda.mem_get_info() free bytes every 5 s
     (min/last) and the failfast/injection counters, so headroom is a measured receipt, not a guess.
  4. FAULT INJECTION (R31B_INJECT="rank=1,after=N", off by default): raise once in DFlash/DSpark propose() on that
     rank after N proposals. Reproduces the desync mechanism deterministically without the OOM trigger.
  5. ACCEPTANCE-GATED K (R31B_KCTL=1, off by default; EngineCore/scheduler process only): replaces the scheduler's
     dynamic-SD lookup with a controller that schedules K=K_OFF (0) while the EMA acceptance length is low and the
     batch-size schedule's K otherwise, probing with full K every R31B_KCTL_PROBE steps while off. Because K is decided
     in the scheduler and broadcast in SchedulerOutput, every rank sees the same K (unlike enable_adaptive_verification,
     whose per-rank confidence state diverges across TP ranks, vllm#54011). Requires --speculative-config to carry
     num_speculative_tokens_per_batch_size (the controller gates that schedule); the worker-side patch of
     cudagraph_utils.build_dynamic_sd_schedule_lookup adds K_OFF's decode graphs on every rank. Without SKIPDRAFT,
     K_OFF is clamped to >= 1 (K=0 broke the DSpark drafter's graph setup in R3.1b).
  6. SKIP-DRAFT (R31C_SKIPDRAFT=1, R3.1c; needs R31B_KCTL=1 and R31B_KCTL_KOFF=0): when the broadcast SchedulerOutput
     says the drafts of this step will be verified at K=0, propose() runs a light step (context K/V only, no draft
     forward, no collective) instead of the draft forward. Target K=0 decode graphs are added; the drafter gets none
     (it never runs at K=0). While OFF the controller probes at full K with exponential back-off (16..R31C_PROBE_MAX)
     and folds each probe with weight 0.5. R31C_CHECK_EVERY=N all-reduces the light-step counters across TP ranks
     every N proposals and fails fast on any disagreement.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys
import threading
import time
import traceback

FAILFAST = os.environ.get("R31B_FAILFAST", "1") == "1"
FAILFAST_METHODS = {"execute_model", "sample_tokens", "execute_dummy_batch"}
FAILFAST_CODE = 86
DRAFT_DG = os.environ.get("R31B_DRAFT_DG_WARMUP", "relax").strip().lower()
STATS_PATH = os.environ.get("R31B_STATS", "")
INJECT = os.environ.get("R31B_INJECT", "").strip()
KCTL = os.environ.get("R31B_KCTL", "0") == "1"
SKIPDRAFT = os.environ.get("R31C_SKIPDRAFT", "0") == "1"
CHECK_EVERY = int(os.environ.get("R31C_CHECK_EVERY", "0"))

_log = lambda m: (sys.stderr.write(f"R31B {m}\n"), sys.stderr.flush())  # noqa: E731
_state = {"failfast": 0, "injected": 0, "proposals": 0, "draft_dg_gemms": 0, "draft_dg_s": None,
          "mem_free_min": None, "mem_free_last": None, "mem_total": None, "kctl": None}


# ---------------------------------------------------------------------------------------------- 1. failfast
def _fatal(method: str, rank, exc: BaseException) -> None:
    _state["failfast"] += 1
    _log(f"FAILFAST rank={rank} method={method}: {exc!r}\n" + "".join(traceback.format_exception(exc)))
    try:
        _dump_stats()
    except Exception:
        pass
    os._exit(FAILFAST_CODE)


def make_execute_worker_rpc(orig, cloudpickle_loads, partial):
    """Return a replacement for WorkerProc._execute_worker_rpc (same contract as af7f9488) that never lets a rank
    swallow an exception from a collective-bearing method and carry on with the next RPC."""

    def _execute_worker_rpc(self, rpc_request):
        method, args, kwargs, output_rank = rpc_request
        name = method if isinstance(method, str) else "<pickled>"
        try:
            if isinstance(method, str):
                func = getattr(self.worker, method)
            else:
                func = partial(cloudpickle_loads(method), self.worker)
            output = func(*args, **kwargs)
            if output_rank is None or self.rank == output_rank:
                self.handle_output(output)
        except Exception as e:
            if FAILFAST and name in FAILFAST_METHODS:
                # Report to the engine first when this is the output rank, so its error names the cause.
                if output_rank is None or self.rank == output_rank:
                    try:
                        self.handle_output(e)
                        time.sleep(0.5)  # async scheduling hands the output to a sender thread
                    except Exception:
                        pass
                _fatal(name, getattr(self, "rank", "?"), e)
            if hasattr(e, "add_note"):
                e.add_note(traceback.format_exc())
            _log(f"WorkerProc exception (not failfast: method={name}): {e!r}\n" + traceback.format_exc())
            if output_rank is None or self.rank == output_rank:
                self.handle_output(e)

    _execute_worker_rpc._r31b = True
    _execute_worker_rpc._r31b_orig = orig
    return _execute_worker_rpc


def _patch_executor(module):
    cls = module.WorkerProc
    if getattr(cls._execute_worker_rpc, "_r31b", False):
        return
    cls._execute_worker_rpc = make_execute_worker_rpc(cls._execute_worker_rpc, module.cloudpickle.loads,
                                                      module.partial)
    _log(f"patched WorkerProc._execute_worker_rpc (failfast={FAILFAST}, methods={sorted(FAILFAST_METHODS)})")


# ------------------------------------------------------------------------------- 2. drafter DeepGEMM warm-up
def draft_m_values(mode: str, max_tokens: int, n: int, device=None) -> list[int]:
    if mode == "full":
        return list(range(1, max_tokens + 1))
    from vllm.model_executor.warmup.deep_gemm_warmup import _generate_optimal_warmup_m_values
    return sorted(_generate_optimal_warmup_m_values(max_tokens, n, device))


def warm_draft_deep_gemm(worker) -> None:
    if DRAFT_DG == "skip":
        return
    runner = getattr(worker, "model_runner", None)
    spec = getattr(runner, "speculator", None)
    model = getattr(spec, "model", None)
    if model is None:
        _log("drafter DeepGEMM warm-up: no draft model (nothing to do)")
        return
    import torch
    from vllm.model_executor.warmup import deep_gemm_warmup as dgw
    from vllm.utils.deep_gemm import fp8_gemm_nt
    max_tokens = worker.scheduler_config.max_num_batched_tokens
    t0 = time.perf_counter()
    shapes, calls = set(), 0
    for mod in model.modules():
        data = dgw._deep_gemm_linear_data(mod)
        if data is None:
            continue
        w, ws = data
        n, k = w.size()
        if (n, k) in shapes:
            continue
        shapes.add((n, k))
        block_m = dgw.get_mk_alignment_for_contiguous_layout()[0]
        a = torch.empty((max_tokens, k), device=w.device, dtype=torch.float8_e4m3fn)
        a_s = torch.zeros((max_tokens, k // block_m), device=w.device, dtype=torch.float32)
        out = torch.empty((max_tokens, n), device=w.device, dtype=torch.bfloat16)
        for m in draft_m_values(DRAFT_DG, max_tokens, n, w.device):
            fp8_gemm_nt((a[:m], a_s[:m]), (w, ws), out[:m])
            calls += 1
        del a, a_s, out
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    _state["draft_dg_gemms"], _state["draft_dg_s"] = calls, round(dt, 2)
    free, total = torch.cuda.mem_get_info()
    _log(f"drafter DeepGEMM warm-up ({DRAFT_DG}): {len(shapes)} shapes {sorted(shapes)}, {calls} GEMMs, {dt:.1f} s; "
         f"free after {free / 2**20:.0f} MiB of {total / 2**30:.1f} GiB")


def _patch_gpu_worker(module):
    orig = module.kernel_warmup
    if getattr(orig, "_r31b", False):
        return

    def kernel_warmup(worker, *args, **kwargs):
        res = orig(worker, *args, **kwargs)
        if kwargs.get("process_local_only"):
            return res
        warm_draft_deep_gemm(worker)  # an exception here fails the boot loudly, which is what we want
        _start_stats()
        return res

    kernel_warmup._r31b = True
    module.kernel_warmup = kernel_warmup
    _log(f"patched gpu_worker.kernel_warmup (drafter DeepGEMM warm-up={DRAFT_DG})")


# ------------------------------------------------------------------------------------------- 3. telemetry
_stats_started = False


def _dump_stats() -> None:
    if not STATS_PATH:
        return
    tmp = STATS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"time": time.time(), **_state, "skip": dict(_skip)}, f)
    os.replace(tmp, STATS_PATH)


def _start_kctl_stats(ctl) -> None:
    """EngineCore process (rank 0's container, which also holds rank 0's worker writing STATS_PATH): the controller's
    state goes to its own file, STATS_PATH with .kctl.json, every 5 s."""
    if not STATS_PATH:
        return
    path = STATS_PATH.rsplit(".json", 1)[0] + ".kctl.json"

    def loop():
        while True:
            try:
                with open(path + ".tmp", "w") as f:
                    json.dump({"time": time.time(), **ctl.snapshot()}, f)
                os.replace(path + ".tmp", path)
            except Exception as exc:
                _log(f"kctl stats failed: {exc!r}")
            time.sleep(5)

    threading.Thread(target=loop, daemon=True, name="r31c-kctl-stats").start()


def _start_stats() -> None:
    global _stats_started
    if _stats_started or not STATS_PATH:
        return
    _stats_started = True
    import torch

    def loop():
        while True:
            try:
                free, total = torch.cuda.mem_get_info()
                _state["mem_free_last"], _state["mem_total"] = free, total
                if _state["mem_free_min"] is None or free < _state["mem_free_min"]:
                    _state["mem_free_min"] = free
                _dump_stats()
            except Exception as exc:
                _log(f"stats failed: {exc!r}")
            time.sleep(5)

    threading.Thread(target=loop, daemon=True, name="r31b-stats").start()


# ------------------------------------------------------------------------------------------ 4. injection
def parse_inject(spec: str) -> dict | None:
    if not spec:
        return None
    kv = dict(p.split("=", 1) for p in spec.split(",") if "=" in p)
    return {"rank": int(kv.get("rank", "1")), "after": int(kv.get("after", "50"))}


def _patch_dflash(module):
    _patch_dflash_skip(module)    # inner wrapper (R3.1c)
    _patch_dflash_inject(module)  # outer wrapper: an injected fault fires on light and full steps alike


def _patch_dflash_inject(module):
    inj = parse_inject(INJECT)
    if inj is None:
        return
    cls = module.DFlashSpeculator
    if getattr(cls.propose, "_r31b", False):
        return
    orig = cls.propose

    def propose(self, *args, **kwargs):
        if not kwargs.get("dummy_run", False):
            _state["proposals"] += 1
            if _state["injected"] == 0 and _state["proposals"] > inj["after"]:
                from vllm.distributed import get_tensor_model_parallel_rank
                if get_tensor_model_parallel_rank() == inj["rank"]:
                    _state["injected"] += 1
                    raise RuntimeError(f"R31B injected fault in propose() on tp rank {inj['rank']} "
                                       f"after {inj['after']} proposals")
        return orig(self, *args, **kwargs)

    propose._r31b = True
    cls.propose = propose
    _log(f"patched DFlashSpeculator.propose with fault injection {inj}")


# --------------------------------------------------------------------------- 6. skip-draft (R3.1c, worker side)
# The scheduler's K for step N (SchedulerOutput.num_spec_tokens_to_schedule, "number of spec tokens to schedule for
# the next step") is exactly the number of drafts that step N's propose() produces for step N+1 to verify: the
# AsyncScheduler sets every request scheduled in step N to K placeholders, and step N+1 verifies those placeholders.
# So when K == 0 the draft forward of step N is dead work. The light step keeps only what later drafts need: the
# drafter's context K/V for this step's target tokens (combine_hidden_states + prepare_dflash_inputs + an eager
# context precompute, none of which has a collective). The decision is a pure function of the broadcast
# SchedulerOutput, so every TP rank skips on exactly the same steps.
_skip = {"next_k": None, "calls": 0, "light": 0, "full": 0, "checks": 0, "async": None}

PROPOSE_PARAMS = ("input_batch", "attn_metadata", "slot_mappings", "last_hidden_states", "aux_hidden_states",
                  "num_sampled", "num_rejected", "last_sampled", "next_prefill_tokens", "temperature", "seeds")


def record_next_k(scheduler_output, dummy_run: bool) -> None:
    """Called at the top of GPUModelRunner.execute_model with the step's broadcast SchedulerOutput."""
    k = None if dummy_run else getattr(scheduler_output, "num_spec_tokens_to_schedule", None)
    _skip["next_k"] = None if k is None else int(k)


def light_step(next_k, dummy_run: bool, is_profile: bool, async_sched: bool) -> bool:
    return bool(SKIPDRAFT and async_sched and not dummy_run and not is_profile and next_k == 0)


def light_propose(spec, prepare_dflash_inputs, a: dict):
    """DFlashSpeculator.propose (af7f9488) minus the draft forward: same hidden-state staging, same per-group
    prepare_dflash_inputs call, then the context K/V precompute for ALL target rows eagerly (the FULL-graph branch
    would have stored the first num_context rows inside the graph we are not replaying)."""
    import torch
    input_batch = a["input_batch"]
    num_reqs = input_batch.num_reqs
    num_target_tokens = input_batch.num_tokens
    with torch.inference_mode():
        aux = a.get("aux_hidden_states")
        if aux:
            hidden_states = spec.model.combine_hidden_states(torch.cat(aux, dim=-1))
        else:
            hidden_states = a["last_hidden_states"]
        spec.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])
        spec.prepare_context_anchor(input_batch, a["num_rejected"])
        if spec.pcp_manager is not None:
            spec.block_tables.gather_block_tables(input_batch.idx_mapping, num_reqs_padded=num_reqs)
        for i, gid in enumerate(spec.draft_kv_cache_group_ids):
            prepare_dflash_inputs(
                spec.input_buffers,
                spec.block_tables.slot_mappings[gid],
                spec.context_positions,
                spec._context_slot_mappings[i],
                spec.sample_indices,
                spec.sample_pos,
                spec.sample_idx_mapping,
                spec.temperature,
                spec.seeds,
                input_batch,
                a["num_sampled"],
                a["num_rejected"],
                a["last_sampled"],
                a["next_prefill_tokens"],
                a["temperature"],
                a["seeds"],
                spec.block_tables.input_block_tables[gid],
                spec.block_tables.kernel_block_sizes[gid],
                spec.block_tables.cp_rank,
                spec.dcp_size,
                spec.block_tables.cp_interleave,
                spec.parallel_drafting_token_id,
                spec.num_query_per_req,
                spec.num_speculative_steps,
                spec.max_num_reqs,
                spec.max_num_tokens,
                spec.max_model_len,
                spec.sample_from_anchor,
            )
        spec._precompute_context_kv(0, num_target_tokens)
    # Never verified: the next step schedules 0 placeholders for every request of this step.
    return spec.draft_tokens[:num_reqs]


def _rank_check(n_every: int) -> None:
    """Every n_every proposals, all-reduce (calls, light) over the TP CPU group and die if the ranks disagree.
    Both ranks reach the same call count on the same step, so the check itself is rank-symmetric."""
    if n_every <= 0 or _skip["calls"] % n_every:
        return
    import torch
    import torch.distributed as dist
    group, rank, world = _tp_cpu_group()
    if world <= 1:
        return
    c, s = _skip["calls"], _skip["light"]
    t = torch.tensor([c, s, -c, -s], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    _skip["checks"] += 1
    if not (t[0] == -t[2] and t[1] == -t[3]):
        _skip["check_failed"] = True
        _fatal("skipdraft-rank-check", rank,
               RuntimeError(f"TP ranks disagree on light steps: calls max {t[0]} min {-t[2]}, "
                            f"light max {t[1]} min {-t[3]}"))


def _tp_cpu_group():
    from vllm.distributed import get_tp_group
    tp = get_tp_group()
    return tp.cpu_group, tp.rank_in_group, tp.world_size


def make_skip_propose(orig, prepare_dflash_inputs, check_every: int = 0):
    def propose(self, *args, **kwargs):
        a = dict(zip(PROPOSE_PARAMS, args))
        a.update(kwargs)
        dummy, profile = bool(a.get("dummy_run", False)), bool(a.get("is_profile", False))
        if dummy or profile:
            return orig(self, *args, **kwargs)
        if _skip["async"] is None:
            _skip["async"] = bool(getattr(self.vllm_config.scheduler_config, "async_scheduling", False))
            if not _skip["async"]:
                _log("SKIPDRAFT: async scheduling is off, so step N's K does not bound step N+1's drafts; disabled")
        _skip["calls"] += 1
        if light_step(_skip["next_k"], dummy, profile, _skip["async"]):
            _skip["light"] += 1
            out = light_propose(self, prepare_dflash_inputs, a)
        else:
            _skip["full"] += 1
            out = orig(self, *args, **kwargs)
        _rank_check(check_every)
        return out

    propose._r31c = True
    return propose


def _patch_dflash_skip(module):
    if not SKIPDRAFT:
        return
    cls = module.DFlashSpeculator
    if getattr(cls.propose, "_r31c", False):
        return
    cls.propose = make_skip_propose(cls.propose, module.prepare_dflash_inputs, CHECK_EVERY)
    _log(f"patched DFlashSpeculator.propose (skip-draft light steps at K=0, rank check every {CHECK_EVERY})")


def _patch_model_runner(module):
    if not SKIPDRAFT:
        return
    cls = module.GPUModelRunner
    orig = cls.execute_model
    if getattr(orig, "_r31c", False):
        return

    def execute_model(self, scheduler_output, *args, **kwargs):
        record_next_k(scheduler_output, bool(kwargs.get("dummy_run", args[1] if len(args) > 1 else False)))
        return orig(self, scheduler_output, *args, **kwargs)

    execute_model._r31c = True
    cls.execute_model = execute_model
    _log("patched GPUModelRunner.execute_model (records the broadcast num_spec_tokens_to_schedule)")


# ------------------------------------------------------------------------------ 5. acceptance-gated K
class AcceptanceGatedK:
    """Scheduler-side K choice. lookup[batch_size] -> K, like vLLM's dense dynamic-SD table.

    The scheduler calls observe() once per finished request-step with (drafted, accepted); __getitem__ is called once
    per schedule() with the batch size. All state lives in the EngineCore process, so the choice is broadcast to
    every rank with the SchedulerOutput.
    """

    def __init__(self, base_lookup: list[int], k_off: int = 0, off_below: float = 1.5, on_above: float = 1.9,
                 alpha: float = 0.2, probe_every: int = 16, min_obs: int = 4, probe_max: int | None = None):
        self.base = list(base_lookup)
        self.k_off = k_off
        self.off_below, self.on_above = off_below, on_above
        self.alpha, self.probe_every, self.min_obs = alpha, probe_every, min_obs
        # R3.1c: with K_OFF=0 the OFF state produces no acceptance observations except at probes, so (a) a probe runs
        # full K long enough to collect min_obs request-steps and its mean alone decides OFF->ON (> on_above), and
        # (b) a failed probe doubles the gap to the next one, up to probe_max (None keeps R3.1b's fixed cadence).
        self.probe_max = probe_every if probe_max is None else max(probe_every, probe_max)
        self.probe_gap = probe_every
        self._since_probe = self._probe_left = 0
        self.ema = None          # EMA of acceptance length (1 + accepted per drafting request-step)
        self.on = True
        self.steps_off = 0
        self._acc = self._n = 0  # accumulated since last schedule()
        self.flips = 0
        self.probes = 0
        self.k_hist: dict[int, int] = {}

    def observe(self, num_draft_tokens: int, num_accepted: int) -> None:
        if num_draft_tokens > 0:
            self._acc += num_accepted
            self._n += 1

    def _fold(self) -> None:
        if not (self._n >= self.min_obs or (self._n > 0 and self.ema is None)):
            return
        al = 1.0 + self._acc / self._n
        self._acc = self._n = 0
        if not self.on and self.k_off == 0:
            # R3.1c OFF state: the only observations are a probe's (>= min_obs request-steps, see _choose), so the
            # probe IS the fresh estimate. ON needs the probe above on_above; OFF needed the EMA below off_below.
            self.ema = al
            if al > self.on_above:
                self.on = True
                self.flips += 1
            else:
                self.probe_gap = min(self.probe_gap * 2, self.probe_max)  # failed probe: back off
            return
        self.ema = al if self.ema is None else (1 - self.alpha) * self.ema + self.alpha * al
        if self.on and self.ema < self.off_below:
            self.on, self.steps_off = False, 0
            self.probe_gap, self._since_probe, self._probe_left = self.probe_every, 0, 0
            self.flips += 1
        elif not self.on and self.ema > self.on_above:
            self.on = True
            self.flips += 1

    def _choose(self, batch_size: int) -> int:
        self._fold()
        k = self.base[batch_size]
        if self.on or k == 0:
            return k
        self.steps_off += 1
        if self._probe_left > 0:
            self._probe_left -= 1
            return k
        self._since_probe += 1
        if self._since_probe >= self.probe_gap:
            # Probe: full K for enough consecutive steps to collect min_obs request-steps (1 step at batch >= min_obs).
            self._since_probe = 0
            self.probes += 1
            if self.k_off == 0:
                self._probe_left = max(1, -(-self.min_obs // max(1, batch_size))) - 1
            return k
        return self.k_off

    def __getitem__(self, batch_size: int) -> int:
        k = self._choose(batch_size)
        self.k_hist[k] = self.k_hist.get(k, 0) + 1
        return k

    def snapshot(self) -> dict:
        return {"on": self.on, "ema": None if self.ema is None else round(self.ema, 3), "flips": self.flips,
                "steps_off": self.steps_off, "probes": self.probes, "probe_gap": self.probe_gap,
                "k_hist": dict(self.k_hist)}


def _patch_scheduler(module):
    if not KCTL:
        return
    cls = module.Scheduler
    if getattr(cls.__init__, "_r31b", False):
        return
    orig_init, orig_stats = cls.__init__, cls.make_spec_decoding_stats

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if self.dynamic_sd_lookup is None:
            _log("KCTL: no num_speculative_tokens_per_batch_size in --speculative-config; controller NOT installed")
            return
        ctl = AcceptanceGatedK(
            self.dynamic_sd_lookup,
            k_off=kctl_k_off(),
            off_below=float(os.environ.get("R31B_KCTL_OFF", "1.5")),
            on_above=float(os.environ.get("R31B_KCTL_ON", "1.9")),
            probe_every=int(os.environ.get("R31B_KCTL_PROBE", "16")),
            probe_max=int(os.environ.get("R31C_PROBE_MAX", "256")) if SKIPDRAFT else None)
        self.dynamic_sd_lookup = ctl
        _state["kctl"] = ctl.snapshot()
        _start_kctl_stats(ctl)
        _log(f"KCTL installed: base={ctl.base[1:]} k_off={ctl.k_off} off<{ctl.off_below} on>{ctl.on_above} "
             f"probe/{ctl.probe_every}..{ctl.probe_max} skipdraft={SKIPDRAFT}")

    def make_spec_decoding_stats(self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
                                 num_invalid_spec_tokens, request_id):
        ctl = self.dynamic_sd_lookup
        if isinstance(ctl, AcceptanceGatedK):
            n = num_draft_tokens - ((num_invalid_spec_tokens or {}).get(request_id, 0))
            ctl.observe(n, num_accepted_tokens)
        return orig_stats(self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
                          num_invalid_spec_tokens, request_id)

    __init__._r31b = True
    cls.__init__ = __init__
    cls.make_spec_decoding_stats = make_spec_decoding_stats
    _log("patched Scheduler (acceptance-gated K)")


# ----------------------------------------------------------------------------------------------- install
class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, target, patch):
        self.target, self.patch = target, patch

    def find_spec(self, name, path, target=None):
        if name != self.target:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig_exec=orig_exec, _patch_fn=self.patch):
            _orig_exec(module)
            _patch_fn(module)

        spec.loader.exec_module = exec_module
        return spec


def _patch_cudagraph_utils(module):
    """KCTL only: make the worker capture decode graphs for K_OFF too. The dense batch-size schedule is used there only
    as a SET of K values ({k + 1 for k in schedule[1:]}); appending K_OFF adds that query length and changes nothing
    else. Same config on every rank, so the captured graph set stays rank-identical."""
    if not KCTL:
        return
    orig = module.build_dynamic_sd_schedule_lookup
    if getattr(orig, "_r31b", False):
        return
    k_off = kctl_k_off()

    def build_dynamic_sd_schedule_lookup(*args, **kwargs):
        dense = orig(*args, **kwargs)
        if k_off in dense[1:]:
            return dense
        if k_off == 0 and _caller_new_tokens_per_step(sys._getframe(1)) == 0:
            # R3.1c: the DSpark drafter's manager (decode_query_len == num_speculative_tokens, 0 new sampled tokens
            # per step) would turn K=0 into a query length of 0 (the R3.1b B2 ZeroDivisionError). The drafter never
            # runs at K=0 under SKIPDRAFT (the step is a light context-only step), so it needs no graph for it.
            return dense
        return dense + [k_off]

    build_dynamic_sd_schedule_lookup._r31b = True
    module.build_dynamic_sd_schedule_lookup = build_dynamic_sd_schedule_lookup
    _log(f"patched cudagraph_utils.build_dynamic_sd_schedule_lookup (+K_OFF={k_off} decode graphs"
         f"{', target only' if k_off == 0 else ''})")


def _caller_new_tokens_per_step(frame) -> int | None:
    """num_new_sampled_tokens_per_step of the CudaGraphManager whose _init_candidates is calling (None if unknown).
    Same formula as af7f9488 cudagraph_utils: decode_query_len - vllm_config.num_speculative_tokens."""
    mgr = frame.f_locals.get("self") if frame is not None else None
    try:
        return int(mgr.decode_query_len) - int(mgr.vllm_config.num_speculative_tokens)
    except Exception:
        return None


def kctl_k_off() -> int:
    """K the controller schedules while OFF. 0 only with SKIPDRAFT (the worker then runs a light context-only step
    instead of a draft forward); otherwise clamped to >= 1 as in R3.1b (K=0 crashed the drafter's graph setup)."""
    k_off = int(os.environ.get("R31B_KCTL_KOFF", "0"))
    if k_off < 1 and not SKIPDRAFT:
        _log(f"KCTL: K_OFF={k_off} unsupported with DSpark graphs unless R31C_SKIPDRAFT=1; using 1")
        k_off = 1
        os.environ["R31B_KCTL_KOFF"] = "1"
    return max(0, k_off)


TARGETS = {
    "vllm.v1.worker.gpu.cudagraph_utils": _patch_cudagraph_utils,
    "vllm.v1.executor.multiproc_executor": _patch_executor,
    "vllm.v1.worker.gpu_worker": _patch_gpu_worker,
    "vllm.v1.worker.gpu.spec_decode.dflash.speculator": _patch_dflash,
    "vllm.v1.worker.gpu.model_runner": _patch_model_runner,
    "vllm.v1.core.sched.scheduler": _patch_scheduler,
}


def install():
    for target, patch in TARGETS.items():
        if target in sys.modules:
            patch(sys.modules[target])
        else:
            sys.meta_path.insert(0, _Finder(target, patch))
