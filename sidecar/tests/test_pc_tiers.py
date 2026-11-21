# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""PC (2026-10-09): CPU tests for the E3 hook's tier counters (_setup_tiers / _count)."""
import importlib.util
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _hook():
    os.environ["E3_COUNT"] = "/tmp/pc-test-counts.json"
    spec = importlib.util.spec_from_file_location("e3_hook_pc_test", os.path.join(HERE, "..", "hook", "e3_hook.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _state(m, rank=0):
    E = m.E
    em = torch.full((E,), -1, dtype=torch.int32)
    em[192 * rank:192 * rank + 192] = torch.arange(192, dtype=torch.int32)
    lo = 192 * rank
    hot = list(range(lo, lo + 100))
    warm = list(range(lo + 100, lo + 140))
    cold = list(range(lo + 140, lo + 192))
    mh = torch.full((E,), -1, dtype=torch.int32)
    mh[hot] = torch.arange(len(hot), dtype=torch.int32)
    mc = torch.full((E,), -1, dtype=torch.int32)
    mc[cold] = torch.arange(len(cold), dtype=torch.int32)       # live map: warm already masked out
    st = m.LayerState(5)
    st.lut = torch.full((E,), -1, dtype=torch.int32)
    st.lut[warm] = torch.arange(len(warm), dtype=torch.int32)
    st.e2 = types.SimpleNamespace(map_hot=mh, map_cold=mc, cold={"w1": torch.zeros(1)}, n_cold=92)
    st.counts = torch.zeros(m.NB, m.NF, dtype=torch.int64)
    m._setup_tiers(st, em, torch.device("cpu"))
    return st, hot, warm, cold


def test_tier_sizes_and_no_overlap():
    m = _hook()
    st, hot, warm, cold = _state(m)
    assert st.tier_info["hot"] == 100 and st.tier_info["warm"] == 40 and st.tier_info["cold"] == 52
    assert st.tier_info["remote"] == 192
    assert st.tier_info["overlap_warm_cold"] == 0 and st.tier_info["local_unassigned"] == 0


def test_count_decode_and_batch():
    m = _hook()
    st, hot, warm, cold = _state(m)
    # T=1 decode: 2 hot, 1 warm, 1 cold, 2 remote
    ids = torch.tensor([[hot[0], hot[1], warm[0], cold[3], 300, 301]], dtype=torch.int32)
    m._count(st, ids, 1)
    acc = st.counts[0]
    assert acc[m.F_TR:m.F_TR + 4].tolist() == [2, 2, 1, 1]
    assert acc[m.F_TU:m.F_TU + 4].tolist() == [2, 2, 1, 1]
    assert acc[m.F_H + 1].item() == 1 and acc[m.F_H:m.F_H + m.HB].sum().item() == 1
    # T=3 batch: the same cold expert twice + another cold one; one invalid (-1) id
    ids = torch.tensor([[cold[0], cold[0], cold[1], hot[0], -1, 200],
                        [hot[0], hot[2], warm[1], 210, 220, 230],
                        [cold[0], hot[3], hot[4], 250, 260, 270]], dtype=torch.int32)
    m._count(st, ids, 3)
    acc = st.counts[1]
    assert acc[m.F_TR:m.F_TR + 4].tolist() == [7, 5, 1, 4]       # 17 valid routes
    assert acc[m.F_TU:m.F_TU + 4].tolist() == [7, 4, 1, 2]       # unique: cold {0,1}, hot {0,2,3,4}
    assert acc[m.F_H + 2].item() == 1
    assert acc[m.F_CNT + cold[0]].item() == 3 and acc[m.F_CNT:m.F_CNT + m.E].sum().item() == 17
    assert st.counts[2].sum().item() == 0
