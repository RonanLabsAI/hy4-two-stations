# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""Shared-buffer layout and handshake under the Hy4 geometry (E3_CONFIG set), in a fresh interpreter each so the
module-level geometry is the one under test."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "models", "hy4", "config.json")

PROBE = r"""
import json, os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "tests"))
import conftest  # triton stub when triton is missing
sys.path.insert(0, os.path.join(sys.argv[1], "hook"))
import peer_tier_v4 as pt
print(json.dumps({"H": pt.HIDDEN, "K": pt.TOPK, "KPAD": pt.KPAD, "MAX_LAYERS": pt.MAX_LAYERS,
  "TABLE_OFF": pt.TABLE_OFF, "SLOT": pt.SLOT, "HEADER": pt.HEADER, "X_OFF": pt.X_OFF, "IDS_OFF": pt.IDS_OFF,
  "W_OFF": pt.W_OFF, "OUT_OFF": pt.OUT_OFF, "TOTAL": pt.TOTAL_BYTES, "MAX_ROWS": pt.MAX_ROWS,
  "SMALL": pt.SMALL_ROWS}))
"""


def probe(env_extra):
    env = dict(os.environ, **env_extra)
    out = subprocess.check_output([sys.executable, "-c", PROBE, ROOT], env=env, text=True)
    return json.loads(out.strip().splitlines()[-1])


def test_hy4_layout():
    d = probe({"E3_CONFIG": CONFIG})
    assert (d["H"], d["K"], d["KPAD"], d["MAX_LAYERS"]) == (6144, 8, 8, 128)
    assert d["TABLE_OFF"] + d["MAX_LAYERS"] * 4 <= d["SLOT"]          # 78 layers fit the warm-count table
    assert d["SLOT"] + d["HEADER"] * 8 <= d["X_OFF"]
    assert d["IDS_OFF"] - d["X_OFF"] == d["MAX_ROWS"] * 6144 * 2
    assert d["W_OFF"] - d["IDS_OFF"] == d["MAX_ROWS"] * 8 * 4 and d["OUT_OFF"] - d["W_OFF"] == d["MAX_ROWS"] * 8 * 4
    assert d["TOTAL"] % 4096 == 0 and d["TOTAL"] >= d["OUT_OFF"] + d["MAX_ROWS"] * 6144 * 2
    assert d["SMALL"] * d["KPAD"] <= 4096


def test_dsv4_layout_unchanged():
    d = probe({"E3_CONFIG": ""})
    assert (d["H"], d["K"], d["KPAD"]) == (7168, 6, 8)
    assert d["X_OFF"] == 8192 and d["IDS_OFF"] == 8192 + 8192 * 7168 * 2


def test_bad_small_rows_rejected():
    env = dict(os.environ, E3_CONFIG=CONFIG, E3_SMALL_ROWS="48")
    r = subprocess.run([sys.executable, "-c", PROBE, ROOT], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "E3_SMALL_ROWS" in r.stderr


def test_rank_warm_hy4_split():
    sys.path.insert(0, os.path.join(ROOT, "hook"))
    import peer_tier_v4 as pt
    rm = {"layers": {"1": {"warm": [0, 127, 128, 255]}, "77": {"warm": [5]}}}
    assert pt.rank_warm(rm, 0, 128) == {1: [0, 127], 77: [5]}
    assert pt.rank_warm(rm, 1, 128) == {1: [128, 255]}
    json.dumps(pt.rank_warm(rm, 1, 128))
