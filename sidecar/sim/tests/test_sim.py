"""Unit checks for sim/n1n2_sim.py (CPU, seconds)."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import n1n2_sim as S  # noqa: E402

C = S.Cost(0.0476, 0.0848, 0.003)


def test_layer_time_base_and_balance():
    assert C.t(0) == 0.003
    # two cold experts on rank 0, none on rank 1
    picks = [(0, True), (0, True)]
    base = S.layer_time(C, picks, mode="base")[0]
    n1 = S.layer_time(C, picks, mode="n1")[0]
    assert abs(base - C.t(2)) < 1e-12 and abs(n1 - C.t(1)) < 1e-12
    # one cold expert: N1 cannot help, N1s splits it
    one = [(0, True)]
    assert abs(S.layer_time(C, one, mode="n1")[0] - C.t(1)) < 1e-12
    assert abs(S.layer_time(C, one, mode="n1s")[0] - C.t(0.5)) < 1e-12
    assert abs(S.layer_time(C, one, mode="n1h")[0] - C.t(0.5)) < 1e-12
    # not replicated -> no move
    assert abs(S.layer_time(C, [(0, False), (0, False)], mode="n1s")[0] - C.t(2)) < 1e-12


def test_modes_never_slower_than_base():
    rng = np.random.default_rng(0)
    for _ in range(300):
        k = rng.integers(1, 6)
        picks = [(int(rng.integers(0, 2)), bool(rng.integers(0, 2))) for _ in range(k)]
        b = S.layer_time(C, picks, mode="base")[0]
        for m in ("n1", "n1s", "n1h"):
            assert S.layer_time(C, picks, mode=m)[0] <= b + 1e-12


def test_simulate_n2_only_reduces_and_zero_eps_is_noop():
    rng = np.random.default_rng(1)
    p = np.full((S.LAYERS, S.E), 1.0 / S.E)
    tier = np.zeros((S.LAYERS, S.E), np.int8)
    tier[:, ::3] = 2
    ids, g = S.gen_routes(p, tier, 1.0, 40, rng)
    rep = np.zeros_like(tier, bool)
    b = S.simulate(ids, g, tier, rep, C, "base")
    z = S.simulate(ids, g, tier, rep, C, "n2", eps=0.0)
    r = S.simulate(ids, g, tier, rep, C, "n2", eps=0.1)
    assert abs(z["crit_mean"] - b["crit_mean"]) < 1e-9 and z["rerouted_per_step"] == 0
    assert r["crit_mean"] <= b["crit_mean"] + 1e-9


def test_replica_budget_respected():
    rng = np.random.default_rng(2)
    p = rng.random((S.LAYERS, S.E))
    tier = np.full((S.LAYERS, S.E), 2, np.int8)
    rep, cap = S.replica_sets(p, tier, 10.0)
    assert rep.sum() == sum(cap["replicas_per_host"])
    assert max(cap["grace_gib_per_host"]) <= 10.0 + 1e-9
    rep_h, cap_h = S.replica_sets(p, tier, 10.0, frac=0.5)
    assert sum(cap_h["replicas_per_host"]) == 2 * sum(cap["replicas_per_host"]) or rep_h.sum() > rep.sum()
