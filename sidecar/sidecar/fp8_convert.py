# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""Block-FP8 (compressed-tensors, E4M3 codes x FP32 scale per 128x128 block) -> b12x MX formats, in plain torch.

Hy4-FP8 ships routed experts as E4M3 with one FP32 scale per 128x128 weight block. b12x's SM120 fused MoE takes
  mxfp4  E2M1 codes, two per byte (element 2i in the LOW nibble), UE8M0 scale per 1x32 K-group   (W4A8-MX, "fp4_e8m0_k32")
  mxfp8  E4M3 codes, one per byte, UE8M0 scale per 1x32 K-group plus one FP32 global per expert (W8A8-MX, "mxfp8_e8m0_k32")
Neither is a re-labelling of block FP8: an arbitrary FP32 block scale is not a power of two, so every weight is
dequantised to FP32 and requantised. MXFP4 is the lossy one (8 magnitudes); MXFP8 costs one extra E4M3 rounding.

Scale choice (mxfp4):
  "ocp"  OCP MX v1.0: shared exponent floor(log2(amax)) - 2, elements saturate at +-6 (values in (6, 8) clip)
  "ceil" smallest power of two with amax / scale <= 6 (no clipping, coarser)
  "mse"  per group, whichever of ocp / ceil reconstructs the group with the lower squared error (default)
All functions run on CPU or GPU tensors; the sidecar converts each layer on the 6000 at load time.
"""
from __future__ import annotations

import torch

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MAX = 448.0
E8M0_MAX_CODE = 247   # b12x's DeepSeek contract clamps UE8M0 codes to 247 (2^120); real expert scales are ~2^-8


def dequant_block_fp8(w: torch.Tensor, scale: torch.Tensor, block=(128, 128)) -> torch.Tensor:
    """[N, K] E4M3 (or its uint8 bytes) x [ceil(N/bn), ceil(K/bk)] FP32 -> FP32 [N, K]."""
    if w.dtype == torch.uint8:
        w = w.view(torch.float8_e4m3fn)
    bn, bk = block
    n, k = w.shape
    s = scale.float().repeat_interleave(bn, 0)[:n].repeat_interleave(bk, 1)[:, :k]
    return w.float() * s


def _e2m1_round(a: torch.Tensor) -> torch.Tensor:
    """Magnitudes (already divided by the group scale, >= 0) -> E2M1 magnitude codes 0..7, round to nearest, ties to
    the even code (the OCP / IEEE convention), saturating at 6."""
    grid = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=a.device)
    mids = (grid[1:] + grid[:-1]) / 2
    code = torch.bucketize(a, mids, right=False)     # a == mid -> lower bucket
    tie = torch.isin(a, mids)
    up = torch.bucketize(a, mids, right=True)
    # at an exact midpoint choose the even code of the two neighbours
    code = torch.where(tie & (up % 2 == 0), up, code)
    return code.clamp_(0, 7)


def _e2m1_decode(code: torch.Tensor) -> torch.Tensor:
    grid = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=code.device)
    mag = grid[(code & 7).long()]
    return torch.where((code & 8) != 0, -mag, mag)


def _group_exponents(wg: torch.Tensor, emax_elem: float):
    """wg [..., G, 32] -> (floor exponent for OCP, ceil exponent), as int32, for elements whose max is emax_elem."""
    amax = wg.abs().amax(-1)
    tiny = torch.finfo(torch.float32).tiny
    safe = amax.clamp_min(tiny)
    e_ocp = torch.floor(torch.log2(safe)).to(torch.int32) - int(torch.floor(torch.log2(torch.tensor(emax_elem))))
    e_ceil = torch.ceil(torch.log2(safe / emax_elem)).to(torch.int32)
    return e_ocp, e_ceil, amax


def _codes_for(wg: torch.Tensor, e: torch.Tensor):
    scale = torch.exp2(e.float()).unsqueeze(-1)
    a = (wg.abs() / scale)
    code = _e2m1_round(a).to(torch.uint8)
    code = torch.where(wg < 0, code | 8, code)
    code = torch.where(code == 8, torch.zeros_like(code), code)     # -0 -> +0
    return code, _e2m1_decode(code) * scale


def quant_mxfp4(w: torch.Tensor, method: str = "mse", chunk_rows: int = 512):
    """FP32 [N, K] (K % 32 == 0) -> (packed uint8 [N, K/2], UE8M0 uint8 [N, K/32])."""
    n, k = w.shape
    if k % 32:
        raise ValueError("K must be a multiple of 32")
    packed = torch.empty(n, k // 2, dtype=torch.uint8, device=w.device)
    scales = torch.empty(n, k // 32, dtype=torch.uint8, device=w.device)
    for r0 in range(0, n, chunk_rows):
        wg = w[r0:r0 + chunk_rows].float().reshape(-1, k // 32, 32)
        e_ocp, e_ceil, amax = _group_exponents(wg, 6.0)
        if method == "ocp":
            e = e_ocp
            code, _ = _codes_for(wg, e)
        elif method == "ceil":
            e = e_ceil
            code, _ = _codes_for(wg, e)
        elif method == "mse":
            c1, d1 = _codes_for(wg, e_ocp)
            c2, d2 = _codes_for(wg, e_ceil)
            pick = ((d2 - wg) ** 2).sum(-1) < ((d1 - wg) ** 2).sum(-1)
            e = torch.where(pick, e_ceil, e_ocp)
            code = torch.where(pick.unsqueeze(-1), c2, c1)
        else:
            raise ValueError(method)
        e = torch.where(amax == 0, torch.full_like(e, -127), e)
        sc = (e + 127).clamp(0, E8M0_MAX_CODE).to(torch.uint8)
        code = code.reshape(-1, k)
        packed[r0:r0 + chunk_rows] = code[:, 0::2] | (code[:, 1::2] << 4)
        scales[r0:r0 + chunk_rows] = sc
    return packed, scales


def dequant_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    lo, hi = packed & 0xF, packed >> 4
    v = torch.stack([_e2m1_decode(lo), _e2m1_decode(hi)], -1).reshape(packed.shape[0], -1)
    return v * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=1)


def quant_mxfp8(w: torch.Tensor, chunk_rows: int = 512):
    """FP32 [N, K] -> (E4M3 bytes uint8 [N, K], UE8M0 uint8 [N, K/32]); global scale 1 (b12x *_global_scales)."""
    n, k = w.shape
    if k % 32:
        raise ValueError("K must be a multiple of 32")
    codes = torch.empty(n, k, dtype=torch.uint8, device=w.device)
    scales = torch.empty(n, k // 32, dtype=torch.uint8, device=w.device)
    for r0 in range(0, n, chunk_rows):
        wg = w[r0:r0 + chunk_rows].float().reshape(-1, k // 32, 32)
        amax = wg.abs().amax(-1)
        # smallest power of two with amax / scale <= 448 (no saturation)
        e = torch.ceil(torch.log2(amax.clamp_min(torch.finfo(torch.float32).tiny) / E4M3_MAX)).to(torch.int32)
        e = torch.where(amax == 0, torch.full_like(e, -127), e)
        q = (wg / torch.exp2(e.float()).unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        codes[r0:r0 + chunk_rows] = q.reshape(-1, k).view(torch.uint8)
        scales[r0:r0 + chunk_rows] = (e + 127).clamp(0, E8M0_MAX_CODE).to(torch.uint8)
    return codes, scales


def dequant_mxfp8(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return codes.view(torch.float8_e4m3fn).float() * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=1)


def quant_act_mxfp8(x: torch.Tensor) -> torch.Tensor:
    """Fake-quantise activations [T, K] to MXFP8 E4M3 per 1x32 (what b12x's A8 mode does on the fly)."""
    t, k = x.shape
    xg = x.float().reshape(t, k // 32, 32)
    amax = xg.abs().amax(-1, keepdim=True)
    e = torch.ceil(torch.log2(amax.clamp_min(torch.finfo(torch.float32).tiny) / E4M3_MAX))
    s = torch.exp2(e)
    return ((xg / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float() * s).reshape(t, k)


def quant_act_fp8_group(x: torch.Tensor, group: int = 128) -> torch.Tensor:
    """Fake-quantise activations [T, K] to FP8 E4M3 with a dynamic FP32 scale per 1 x group (vLLM's block-FP8 W8A8
    per-token-group quant: scale = amax / 448)."""
    t, k = x.shape
    xg = x.float().reshape(t, k // group, group)
    s = (xg.abs().amax(-1, keepdim=True) / E4M3_MAX).clamp_min(1e-10)
    return ((xg / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float() * s).reshape(t, k)


def convert_expert(w1, s1, w3, s3, w2, s2, fmt: str = "mxfp4", block=(128, 128), method: str = "mse"):
    """One Hy4 expert -> b12x tensors. Returns dict(w13, s13, w2, s2) with w13 = [gate; up] rows (b12x layout "w31").

    mxfp4: w13 uint8 [2I, H/2], s13 uint8 [2I, H/32], w2 uint8 [H, I/2], s2 uint8 [H, I/32]
    mxfp8: w13 uint8 [2I, H],   s13 uint8 [2I, H/32], w2 uint8 [H, I],   s2 uint8 [H, I/32]
    """
    q = (lambda w: quant_mxfp4(w, method)) if fmt == "mxfp4" else quant_mxfp8
    g, gs = q(dequant_block_fp8(w1, s1, block))
    u, us = q(dequant_block_fp8(w3, s3, block))
    d, ds = q(dequant_block_fp8(w2, s2, block))
    return {"w13": torch.cat([g, u]), "s13": torch.cat([gs, us]), "w2": d, "s2": ds}


def dequant(fmt: str, codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return dequant_mxfp4(codes, scales) if fmt == "mxfp4" else dequant_mxfp8(codes, scales)
