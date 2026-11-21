# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""MoE geometry of the served checkpoint, read from its config.json instead of hard-coded DeepSeek-V4 constants.

Both sides of the E3 protocol (the vLLM worker hook and the RTX PRO 6000 sidecar) import this, so they agree on the
hidden size, top-k, expert count and layer count of the model. Without a config (E3_CONFIG unset) the geometry is
DeepSeek-V4-Pro-0813's, so the DS-V4 kit keeps working unchanged.

  E3_CONFIG=/model/config.json   (the hook and the sidecar both read it; the launchers set it)

Families:
  deepseek_v4  DeepSeek-V4-Pro / V4.1-Flash: experts at layers.N.ffn.experts.E.{w1,w3,w2}.{weight,scale}, packed MXFP4
  hy_v4        Tencent Hy4 (model_type hy_v4 / hy_v4_vl): experts at model.layers.N.mlp.experts.E.
               {gate_proj,up_proj,down_proj}.{weight,weight_scale}, compressed-tensors block FP8 (128x128, FP32 scales)
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Geometry:
    family: str
    model_type: str
    hidden: int
    inter: int                     # moe_intermediate_size (per routed expert)
    n_experts: int                 # routed experts per MoE layer
    topk: int
    n_layers: int                  # decoder layers (MoE + dense), excluding drafter layers
    first_dense: int               # first_k_dense_replace: layers [0, first_dense) have no routed experts
    n_shared: int = 0
    swiglu_limit: float | None = None
    routed_scaling: float = 1.0
    norm_topk_prob: bool = True
    weight_format: str = "mxfp4"   # checkpoint expert format: mxfp4 (DS-V4) | fp8_block (Hy4)
    weight_block: tuple = (128, 128)
    extra: dict = field(default_factory=dict)

    @property
    def moe_layers(self) -> list[int]:
        return list(range(self.first_dense, self.n_layers))

    def expert_names(self, layer: int, expert: int) -> dict[str, str]:
        """Checkpoint tensor names of one routed expert: keys w1 (gate), w3 (up), w2 (down) and their scales."""
        if self.family == "hy_v4":
            p = f"{self.extra.get('prefix', 'model.')}layers.{layer}.mlp.experts.{expert}"
            return {"w1": f"{p}.gate_proj.weight", "w3": f"{p}.up_proj.weight", "w2": f"{p}.down_proj.weight",
                    "w1s": f"{p}.gate_proj.weight_scale", "w3s": f"{p}.up_proj.weight_scale",
                    "w2s": f"{p}.down_proj.weight_scale"}
        p = f"layers.{layer}.ffn.experts.{expert}"
        return {"w1": f"{p}.w1.weight", "w3": f"{p}.w3.weight", "w2": f"{p}.w2.weight",
                "w1s": f"{p}.w1.scale", "w3s": f"{p}.w3.scale", "w2s": f"{p}.w2.scale"}

    def expert_bytes(self, fmt: str | None = None) -> int:
        """Bytes of one routed expert (w1 + w3 + w2 and scales) in `fmt` (default: the checkpoint's format)."""
        fmt = fmt or self.weight_format
        n = 3 * self.inter * self.hidden
        if fmt == "mxfp4":
            return n // 2 + n // 32
        if fmt == "mxfp8":
            return n + n // 32
        if fmt == "fp8_block":
            bn, bk = self.weight_block
            blocks = 2 * (self.inter // bn) * (self.hidden // bk) + (self.hidden // bn) * (self.inter // bk)
            return n + 4 * blocks
        if fmt == "bf16":
            return 2 * n
        raise ValueError(fmt)

    def to_json(self) -> dict:
        return asdict(self)


DSV4_PRO = Geometry(family="deepseek_v4", model_type="deepseek_v4", hidden=7168, inter=3072, n_experts=384, topk=6,
                    n_layers=61, first_dense=0, n_shared=1, swiglu_limit=10.0, routed_scaling=2.5,
                    norm_topk_prob=True, weight_format="mxfp4", weight_block=(1, 32))


def _text(cfg: dict) -> dict:
    """Merged view: the VL wrapper (hy_v4_vl) keeps the language model's keys both at top level and in text_config;
    text_config wins where both exist."""
    t = dict(cfg)
    t.update(cfg.get("text_config") or {})
    return t


def from_config(cfg: dict) -> Geometry:
    t = _text(cfg)
    mt = t.get("model_type", "")
    top_mt = cfg.get("model_type", mt)
    if mt.startswith("hy_v4") or top_mt.startswith("hy_v4"):
        q = cfg.get("quantization_config") or t.get("quantization_config") or {}
        block = (128, 128)
        fmt = "bf16"
        for g in (q.get("config_groups") or {}).values():
            w = g.get("weights") or {}
            if w.get("type") == "float" and w.get("num_bits") == 8 and w.get("strategy") == "block":
                block, fmt = tuple(w.get("block_structure", [128, 128])), "fp8_block"
        mlp_types = t.get("mlp_layer_types")
        first_dense = int(t.get("first_k_dense_replace", 0))
        if mlp_types:
            dense = [i for i, k in enumerate(mlp_types) if k != "sparse"]
            if dense != list(range(len(dense))):
                raise ValueError(f"non-leading dense layers {dense} are not supported")
            first_dense = len(dense)
        extra = {"prefix": "model.", "hc_mult": t.get("hc_mult"), "enable_ihc": t.get("enable_ihc"),
                 "dfly_num_hidden_layers": cfg.get("dfly_num_hidden_layers"), "dfly_mlp_type": cfg.get("dfly_mlp_type"),
                 "architectures": cfg.get("architectures")}
        return Geometry(family="hy_v4", model_type=top_mt, hidden=int(t["hidden_size"]),
                        inter=int(t["moe_intermediate_size"]), n_experts=int(t["n_routed_experts"]),
                        topk=int(t["num_experts_per_tok"]), n_layers=int(t["num_hidden_layers"]),
                        first_dense=first_dense, n_shared=int(t.get("n_shared_experts", 0) or 0),
                        swiglu_limit=t.get("swiglu_limit"), routed_scaling=float(t.get("routed_scaling_factor", 1.0)),
                        norm_topk_prob=bool(t.get("norm_topk_prob", True)), weight_format=fmt, weight_block=block,
                        extra=extra)
    if mt.startswith("deepseek_v4") or top_mt.startswith("deepseek_v4"):
        return Geometry(family="deepseek_v4", model_type=top_mt, hidden=int(t["hidden_size"]),
                        inter=int(t["moe_intermediate_size"]), n_experts=int(t["n_routed_experts"]),
                        topk=int(t["num_experts_per_tok"]), n_layers=int(t["num_hidden_layers"]),
                        first_dense=int(t.get("first_k_dense_replace", 0) or 0),
                        n_shared=int(t.get("n_shared_experts", 0) or 0), swiglu_limit=t.get("swiglu_limit", 10.0),
                        routed_scaling=float(t.get("routed_scaling_factor", 1.0)),
                        norm_topk_prob=bool(t.get("norm_topk_prob", True)), weight_format="mxfp4",
                        weight_block=(1, 32))
    raise ValueError(f"unsupported model_type {top_mt!r}/{mt!r}")


def load(path: str | None = None) -> Geometry:
    path = path if path is not None else os.environ.get("E3_CONFIG", "")
    if not path:
        return DSV4_PRO
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    with open(path) as f:
        return from_config(json.load(f))


if __name__ == "__main__":
    import sys

    g = load(sys.argv[1] if len(sys.argv) > 1 else None)
    print(json.dumps(g.to_json(), indent=1))
    print({f: round(g.expert_bytes(f) / 1e6, 3) for f in ("fp8_block", "mxfp8", "mxfp4", "bf16")}, "MB per expert")
