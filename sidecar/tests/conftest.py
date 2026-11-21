# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""Test setup: if triton is not importable (a CPU-only venv), install a stub so hook/peer_tier_v4.py imports for the
layout / handshake tests. The stub cannot run kernels; GPU tests skip themselves."""
import sys
import types

try:
    import triton  # noqa: F401
except ImportError:
    tr = types.ModuleType("triton")
    tl = types.ModuleType("triton.language")

    def jit(fn=None, **kw):
        if fn is None:
            return lambda f: f
        return fn

    tr.jit = jit
    tr.cdiv = lambda a, b: (a + b - 1) // b
    tl.constexpr = object
    tr.language = tl
    tr.__stub__ = True
    sys.modules["triton"] = tr
    sys.modules["triton.language"] = tl
