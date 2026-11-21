# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""CPU tests for the Hy4 placement model and memory budget (tools/build_rowmap.py, tools/hy4_budget.py)."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hook"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_rowmap as br  # noqa: E402
import geometry  # noqa: E402
import hy4_budget as hb  # noqa: E402

import pytest  # noqa: E402

G = geometry.load(os.path.join(ROOT, "models", "hy4", "config.json"))


@pytest.fixture(autouse=True)
def _restore_geometry():
    """build_rowmap keeps its geometry in module globals (DS-V4-Pro by default); put it back after each test."""
    yield
    br.set_geometry(geometry.DSV4_PRO)


def test_offload_flag_gives_whole_layers():
    """The flag must offload whole layers only: no trailing w13 of the next layer (vLLM counts weights, not scales)."""
    br.set_geometry(G)
    for ep, n_local in ((2, 128), (1, 256)):
        for k in (1, 10, 37, 59, 76):
            flag = hb.offload_flag(k, n_local)
            trace = br.offload_trace(flag, ep)
            assert trace == [(l, p) for l in range(1, 1 + k) for p in ("w13", "w2")], (ep, k, flag, trace[-3:])
    assert hb.offload_flag(37, 128) == 166.5 and hb.offload_flag(38, 128) == 171.0
    assert hb.offload_flag(59, 256) == 531.0
    # the pre-fix value 166.54 (scales counted) pulls in layer 38's w13 -- the bug this test guards
    assert br.offload_trace(166.54, 2)[-1] == (38, "w13")


def test_budget_verdicts():
    b4 = hb.budget(G, offload_gib=0, warm_fmt="mxfp4")
    b8 = hb.budget(G, offload_gib=0, warm_fmt="mxfp8")
    assert b4["B"]["hbm_fits"] and b4["B"]["grace_fits"] and b4["B"]["n_grace_layers"] == 37
    assert b4["A"]["rtx_fits"] and b4["A"]["warm_experts"] == 4283 and b8["A"]["warm_experts"] == 2206
    assert not b4["single"]["grace_full_copy_fits"]
    assert b4["single"]["grace_exclusive_fits"] and not b8["single"]["grace_exclusive_fits"]


def test_prior_rowmaps_match_budget_and_placement():
    for fmt, n in (("mxfp4", 4283), ("mxfp8", 2206)):
        rm = json.load(open(os.path.join(ROOT, "rowmaps", f"rowmap-hy4-prior-{fmt}.json")))
        assert rm["geometry"]["family"] == "hy_v4" and rm["geometry"]["warm_fmt"] == fmt
        assert rm["placement"]["grace_layers"] == list(range(1, 38)) and rm["placement"]["offload_gib"] == 166.5
        for rank in (0, 1):
            ids = [e for v in rm["layers"].values() for e in v["warm"] if rank * 128 <= e < rank * 128 + 128]
            assert len(ids) == n
        for v in rm["layers"].values():
            for rank in (0, 1):
                k = sum(rank * 128 <= e < rank * 128 + 128 for e in v["warm"])
                assert 1 <= k <= 127                         # >= 1 cold expert per layer per rank


def test_rowmap_from_counts(tmp_path):
    """COUNT-run output -> rowmap: the most-routed local experts of the Grace layers become warm."""
    counts = {"format": "routing-counts/v1+pc-tiers", "layers": {}}
    for layer in range(1, 78):
        c = [1] * 256
        c[3] = c[200] = 1000                                  # one hot expert per rank in every layer
        counts["layers"][str(layer)] = c
    cp = tmp_path / "counts.json"
    cp.write_text(json.dumps(counts))
    out = tmp_path / "rm.json"
    subprocess.check_call([sys.executable, os.path.join(ROOT, "tools", "build_rowmap.py"), "--config",
                           os.path.join(ROOT, "models", "hy4", "config.json"), "--counts", str(cp), "--offload-gib",
                           "166.5", "--warm-fmt", "mxfp8", "--capacity", "370", "--min-per-layer", "8", "--out",
                           str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rm = json.load(open(out))
    assert sorted(int(k) for k in rm["layers"]) == list(range(1, 38))
    for v in rm["layers"].values():
        assert 3 in v["warm"] and 200 in v["warm"]
    assert rm["source"]["prior"] is None


def test_counts_delta_removes_boot_traffic():
    import counts_delta as cd
    before = {"time": 1, "layers": {"1": [8192, 0, 5], "2": [0, 0, 0]}, "by_bucket": {"2": {"1": [8192, 0, 5]}}}
    after = {"time": 2, "layers": {"1": [8200, 40, 5], "2": [3, 0, 1]}, "by_bucket": {"2": {"1": [8195, 40, 5]}}}
    d = cd.delta(before, after)
    assert d["layers"] == {"1": [8, 40, 0], "2": [3, 0, 1]} and d["by_bucket"]["2"]["1"] == [3, 40, 0]
    import pytest as _pt
    with _pt.raises(ValueError):
        cd.delta(after, before)
