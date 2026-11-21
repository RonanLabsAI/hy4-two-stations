# Hy4 warm-expert sidecar (port of the DS-V4 E3 kit)

Branch `feat/hy4-sidecar`, cut from dsv4pro-sidecar `b09fb2d`. This is a port of the RTX PRO 6000 warm-expert sidecar from
DeepSeek-V4 to Tencent Hy4 (`tencent/Hy4-FP8` @ `d063b5853ecf56e0776266dab28305fc17a137cc`). The target engine is
`vllm/vllm-openai:hy4` (index digest `sha256:d21fb990…`): the arm64 digest is `sha256:e2682777489d…`, built from `thisjiang/vllm@5790aa815a18` plus
Tencent hpc-ops `4c3b3c98`.

Everything here is desk work (2026-11-20). Nothing in this branch has run on a GPU. The derivation and license
chain is unchanged: Jason Cook's Apache-2.0 code, see `NOTICE` and the per-file headers.

## What changed

| File | Change |
|---|---|
| `hook/geometry.py` (new) | Reads the MoE geometry from `config.json` (`E3_CONFIG`). Families: `deepseek_v4` (the default when unset) and `hy_v4`. Hy4: H 6144, I 2048, 256 routed + 1 shared expert, top-8, 78 layers (layer 0 dense), swiglu_limit 10, routed_scaling 2.827, block FP8 128x128. |
| `hook/peer_tier_v4.py` | Takes H and top-k from the geometry. Warm-count table grows to 128 layers. The publish-kernel padding is sized from top-k. A geometry handshake (words 11/12) makes the worker refuse a sidecar built for another model. |
| `hook/e3_hook.py` | Geometry-driven. Adds the Hy4 layer names (`[language_model.]model.layers.N.mlp.experts`). Drafter (dfly / mtp) layers are never hooked. Adds `E3_MODE=count` (routing counts without a sidecar or rowmap). The cosine reference runs without shared experts. `E3_ALLOW_NO_EP=1` installs an identity expert map for TP1. |
| `sidecar/fp8_convert.py` (new) | Converts block FP8 to MXFP4 (E2M1 + UE8M0 per 32, MSE scale choice) or MXFP8 (E4M3 + UE8M0 per 32), with dequant helpers and activation fake-quant. Pure torch. |
| `sidecar/peer_server_v4.py` | Geometry from `E3_CONFIG` and a dtype-aware reader. Hy4 experts are converted on the 6000 at load (`--wfmt mxfp4\|mxfp8`). |
| `tools/offline_gate.py` (new) | Offline per-layer cosine gate on CPU. It compares the 6000's emulated output with the GB300's emulated block-FP8 W8A8 output. |
| `tools/build_rowmap.py` | Takes `--config`, `--warm-fmt` and `--capacity-gib`. Offload modelling covers MoE layers only. Without counts it uses a uniform prior. |
| `tools/hy4_budget.py` (new) | Memory budget for B, A and single-Station layouts. |
| `tools/cos_gate.py` | Adds a `MIN_COS` threshold. |
| `tools/counts_delta.py` (new) | Routing counts between two snapshots (removes boot traffic from a COUNT run). |
| `tools/fetch_experts.py`, `tools/fetch_norms.py` (new) | Fetch real expert tensors from HF by range request, for the CPU fixtures. |
| `launch/hy4/` (new) | `launch-hy4.sh` (recipes B / COUNT / A), `hy4-up.sh`, `hy4-down.sh`, `hy4-sidecar.sh`, `bench-hy4.sh`, `launch-hy4-single.sh` (stretch; refuses to start). |
| `rowmaps/rowmap-hy4-prior-{mxfp4,mxfp8}.json` | Uniform-prior rowmaps for the first A smoke (Grace layers 1-37, 80 GiB of warm experts per 6000). |
| `models/hy4/config.json` | Copy of the checkpoint's config. Its sha256 `2ae591dd…` matches `/models/Hy4-FP8/config.json` on station-right. |

## Hook point in `vllm/vllm-openai:hy4`

The hook point is the same as on DS-V4: `vllm.model_executor.layers.fused_moe.routed_experts.RoutedExperts.forward_modular(x,
topk_weights, topk_ids, shared_experts, shared_experts_input)`. Read from the exact source of the image's commit
(`thisjiang/vllm@5790aa8`):

- **Model.** `vllm/models/hy_v4/nvidia/moe.py:HYV4MoEFused` builds its experts with `FusedMoEFactory(...,
  scoring_func="sigmoid", use_grouped_topk=True, num_expert_group=1, e_score_correction_bias,
  routed_scaling_factor=2.827, shared_experts=HYV4FeedForward, swiglu_limit=10.0)`. The factory returns a `MoERunner`;
  the expert weights live in `RoutedExperts`. The expert layer's name is `<prefix>.mlp.experts`, and its parameters are
  `...mlp.experts.routed_experts.w13_weight` / `w2_weight`.
- **routed_scaling.** It defaults to `RoutedScaleMode.ROUTER`: both possible routers (`GroupedTopKRouter`, or
  `FusedTopKBiasRouter` when grouping degenerates) multiply `topk_weights` by 2.827. The sidecar uses the same scaled
  weights, so `masked + warm == full` holds with no extra factor.
- **Shared expert.** The runner runs it itself (`_maybe_apply_shared_experts`) and returns its output separately from
  `forward_modular`. `forward_modular` returns the routed sum only, so the warm add touches only the routed sum.
- **iHC (hc_mult 4).** It wraps the MLP outside the MoE: `hc_mlp_layer.pre` → `post_attention_layernorm` →
  (sequence-parallel all-gather when T ≥ 1024) → `mlp` → (reduce-scatter / all-reduce) → `hc_mlp_layer.post`.
  The hook sees a plain BF16 `[T, 6144]` input with all tokens on each rank. Each rank's output is a partial sum that
  the deferred reduction adds up, and adding the rank's own warm contribution before that reduction is correct.
- **dfly drafter.** It has 5 layers with a dense MLP (`dfly_mlp_type: dense`, `HYV4FeedForward`), so it has no
  `RoutedExperts` and is never hooked. The regex also rejects `dfly`/`mtp`/`draft` names. It reuses target
  KV/indexer at layers 61-77, which does not touch the MoE.
- **Backend.** The FP8 method (`compressed_tensors_moe_w8a8_fp8.py`) passes `expert_map=layer.expert_map` on every
  `apply`, so swapping `_expert_map` works. The auto backend on SM100-family is `FLASHINFER_TRTLLM`, which is monolithic
  (no `forward_modular`), and its load-time shuffle allocates new GPU tensors, which would undo the UVA offload. The
  kit therefore pins `--moe-backend triton`. Triton accepts `(kFp8Static128BlockSym, kFp8Dynamic128Sym)` on any CUDA
  device, keeps weights in place (`convert_to_fp8_moe_kernel_format` does nothing for TRITON), skips masked routes
  (`ignore_invalid_experts` + `moe_sum(expert_map)`), has a UVA-aware assignment (`weights_host_resident`), and applies
  the same clamp `silu(min(g, 10)) * clamp(u, -10, 10)` (`_swiglu_limit_torch`).
- **Offload.** `--cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight`. The selector matches whole
  dotted segments, so `experts.w13_weight` matches nothing, as on DS-V4. The budget is checked before each parameter
  in module order, so `--cpu-offload-gb 166.5` offloads exactly MoE layers 1-37 (`tools/hy4_budget.py offload_flag`).

## Warm-expert weight path and numerics

Each Hy4 expert is E4M3 with an FP32 scale per 128x128 block. The FP32 scales are not powers of two, so no b12x
format is a relabelling: every warm expert is dequantised and requantised (`sidecar/fp8_convert.py`, on the 6000 at
load).

Offline gate on real Hy4 experts: layers 1/20/40/60/77, experts 0-7 each, fetched from HF at the pinned revision.
Inputs are N(0,1) times the layer's `post_attention_layernorm` weight, with top-8 random routes, 64 tokens per layer
(`calibration/hy4/offline-gate-*.json`). Column definitions:

- **cos(6000, GB300)** is the 6000's emulated output against the GB300's emulated block-FP8 W8A8 output (FP8 per-128
  dynamic activation quant).
- **cos(6000, exact) / cos(GB300, exact)** compare against FP32 activations times dequantised FP8 weights, i.e. the
  checkpoint's own numbers.

| 6000 weights | cos(6000, GB300) per layer | rel. error | cos(6000, exact) | cos(GB300, exact) | weight cos (w13) |
|---|---|---|---|---|---|
| MXFP4 W4A8 (MSE scales) | 0.9787-0.9870 | 16-21% | 0.9796-0.9877 | 0.9991-0.9993 | 0.9936 |
| MXFP8 W8A8 | 0.9969-0.9979 | 6-8% | 0.9979-0.9986 | same | 0.99965 |

- **MXFP4 is a real numerics change.** It is about 20 times the error energy of the GB300's own FP8 path. That is
  inherent to E2M1 (8 magnitudes); MSE scale choice vs OCP gains only about 0.0015 cos (layer 40: 0.9797 vs 0.9782). DS-V4 did not have this
  problem because its checkpoint is MXFP4, so warm and HBM numerics were the same there. The error applies to the warm
  share of routed output only (hot HBM experts, cold Grace experts and the shared expert stay FP8 or BF16), but at
  the prior rowmap the 6000 serves about 90% of Grace-layer experts.
- **Is there a usable FP8 path in b12x on SM120?** Yes, MXFP8 W8A8. b12x master `2cc7f66a`, which is on both Stations
  at `~/dsv4pro-e3/b12x`, has `PackedSourceFormat.MXFP8_E8M0_K32` → recipe `w8a8_mx`: E4M3 codes kept byte-exact,
  UE8M0 per-32 K scales, a per-expert FP32 alpha, the m16n8k32 MXF8 MMA on SM120/121, and A8 activations only. Its own
  GPU tests (`tests/moe/test_mxfp8_w8a8.py`) cover the oracle, routeless experts and CUDA-graph replay. It needs
  hidden % 128 == 0 and inter % 32 == 0, which Hy4 meets. It has no plain block-FP8 (128x128 FP32-scale) source format.
  Costs and limits:
  - Memory doubles: 37.1 vs 19.1 MiB per expert. That gives 2,206 instead of 4,283 warm experts per 6000 (47% vs 90%
    of Grace-layer experts).
  - It cannot fit the single-Station layout (below).
  - The sidecar implements both formats: `--wfmt mxfp8` / `WFMT=mxfp8`, using `rowmap-hy4-prior-mxfp8.json`.
- **GPU gate thresholds.** The DS-V4 bar of 0.999 cannot hold, because the GB300 reference and the sidecar no longer
  share weights. Use `MIN_COS=0.97` for MXFP4 and `0.995` for MXFP8. A layout or scale bug shows up as a cosine near 0,
  or NaN, not 0.96.

## Launch configurations (written, not run)

All recipes are two-node vLLM (mp backend, `--nnodes 2`). Each Station is one EP rank with TP2 attention over rail 1
(mlx5_1, 192.168.101.x). They all use the vendor flags (`--dtype bfloat16 --reasoning-parser hy_v4
--tool-call-parser hy_v4 --attention-backend FLASHMLA_SPARSE --speculative-config
'{"num_speculative_tokens":5,"method":"dfly"}'`) plus `--enable-expert-parallel --moe-backend triton --offload-backend uva
--cpu-offload-gb 166.5`.

- **B (`hy4-up.sh B`):** GB300 holds MoE layers 38-77; Grace holds layers 1-37 (UVA).
- **COUNT (`hy4-up.sh COUNT`):** B plus `E3_MODE=count`. Per-layer, per-rank route histograms are written to
  `LOGDIR/e3-counts-rank{0,1}.json`. They include tier counters (HBM vs Grace routes) and need no sidecar. This is the
  capture run for the first Hy4 rowmap. The counters run from boot, so they include vLLM's dummy profile run (8,192
  identical tokens) and the warm-ups. `hy4-up.sh COUNT` therefore snapshots them at ready, and the rowmap is built
  from the delta:
  `counts_delta.py LOGDIR/e3-counts-ready-rank0.json LOGDIR/e3-counts-rank0.json LOGDIR/delta-rank0.json` then
  `build_rowmap.py --config models/hy4/config.json --counts LOGDIR/delta-rank0.json --offload-gib 166.5
  --warm-fmt mxfp4 --capacity-gib 80 --out rowmaps/rowmap-hy4-v1.json`. Both ranks see the same tokens, so one rank's
  delta is enough. Random-token bench traffic routes differently from real text, so capture on the target workload.
- **A (`hy4-up.sh A`):** B plus the 6000 warm tier. `hy4-sidecar.sh 0` on station-left and `hy4-sidecar.sh 1` on
  station-right must be serving first, with the same `E3ROWMAP`/`WFMT`. The sidecar runs in the `nightly-af7f9488`
  image, which is known-good for b12x 2cc7f66a. It talks to the worker only through `/dev/shm/e3_hy4_peer`.
- **The dfly drafter is on in all three** (`SPEC=5`; `SPEC=0` turns it off for an A/B of the drafter itself).

Memory budget per EP rank, in GiB (`tools/hy4_budget.py`; `calibration/hy4/budget-*.md`). Measured inputs: GB300
256,703 MiB, RTX 97,887 MiB, Grace MemAvailable about 439 GiB. Rows marked "est." are estimates that the first
boot's logs replace.

| per EP rank (GiB) | B | A (mxfp4) | single TP1, A's full-copy mechanism | single TP1, exclusive warm tier |
|---|---|---|---|---|
| --cpu-offload-gb | 166.5 | same | 531.0 | needs a new loader |
| MoE layers in HBM / Grace | 40 / 37 (1-37) | same | 18 / 59 | same |
| HBM: non-expert 22 (est.) + hot + KV 20 (est.) + work 14 (est.) | 236.1 of 238.2 (fits, thin) | same | 235.2 of 238.2 | same |
| 6000: warm + reserve 15.6 (est.) | - | 4,283 experts (80 GiB) | 4,283 experts | same |
| Grace: cold pinned + host 30 (est.) | 196.5 of 439 | 196.5 (full copy kept) | **561.1 of 439: DOES NOT FIT** | 410.5 of 439 (fits with mxfp4, 28.5 margin; mxfp8 483.6: does not fit) |

**Single Station (stretch).** All 77 layers' experts are 693 GiB of FP8, which is more than GB300 + Grace (about
689 GiB usable). It can only fit if the 6000 holds warm experts exclusively, in MXFP4, and Grace drops them. That
needs a loader that allocates compact per-layer cold tensors and remaps the expert map. It is not implemented.
`launch-hy4-single.sh` documents the layout and refuses to start (`FORCE_SINGLE=1` overrides, and it would OOM while
pinning). The hook side is ready (`E3_ALLOW_NO_EP=1`, CPU-tested).

## Verified here (CPU only, no GPU anywhere)

All of these pass: `~/.venvs/hy4-cpu/bin/python -m pytest tests/` gives 62 passed, 5 skipped (GPU-only or
fixture-gated). With `HY4_INDEX` and `HY4_EXPERTS` set, the two fixture-gated Hy4 tests also run: 64 passed, 3 skipped (the 3 are the original DS-V4 GPU / vLLM-source tests).

- **Geometry.** Parsed from the real config. Expert tensor names exist in the real index (`test_geometry_vs_index_names`).
  77 x 256 FP8 experts = 744.3 GB, which matches the checkpoint (786.3 GB total).
- **Converter.**
  - E2M1 round-trip, nibble order (element 2i in the low nibble) and ties-to-even.
  - MSE scales are never worse than OCP or ceil.
  - MXFP8 is near-lossless for power-of-two block scales.
  - The `[gate; up]` ("w31") row order and shapes match b12x's contract.
- **Offline cosine gate on real Hy4 experts** (table above).
- **Sidecar loader.** Dtype-aware safetensors reads (F8_E4M3, F32) and the per-layer conversion shapes for both
  formats, with b12x stubbed.
- **Hook, against a fake `RoutedExperts` implementing the image's contract:**
  - Layer-name parsing (target vs dense layer 0 vs drafter).
  - Count mode needs no sidecar and leaves the map untouched; the counts and tier totals are right.
  - In serve mode, the worker's map masks exactly the rank's warm experts, and wire ids are sorted positions.
  - `masked + exact sidecar == stock` on both EP ranks.
  - The no-EP identity map.
- **Protocol layout under the Hy4 geometry.** 78 layers fit the table, buffer offsets are right, `E3_SMALL_ROWS`
  is validated, and the DS-V4 layout is unchanged.
- **Placement and budget.** `--cpu-offload-gb 166.5` gives exactly layers 1-37 under vLLM's rule. The budget
  verdicts hold, the prior rowmaps match the placement (at least 1 cold expert per layer per rank), and counts lead to
  the right warm picks.
- **The original DS-V4 tests still pass.**
- **vLLM facts above** come from the source at the image's commit, not from running it.

## Only a GPU run can verify

1. b12x actually accepts the converted tensors and matches the emulation (`tests/b12x_layer_check_hy4.py`, cos(b12x, emu)
   ≥ 0.999). That covers layout "w31", UE8M0 clamp 247, the swiglu_limit semantics, and MXFP8 global alpha = 1.
2. The hy4 image starts with `--moe-backend triton` + UVA offload + EP2 + dfly + FLASHMLA_SPARSE on SM103. It must log
   `Total CPU offloaded parameters: 166.5` (vLLM prints GiB without a unit; 169.5 would mean a partial layer 38) on both ranks and fit HBM. The HBM margin estimate is 2 GiB; if KV or
   load OOMs, use `OFFGB=171.0` (38 layers).
3. Triton block-FP8 throughput on GB300 (B's baseline will be lower than the vendor TRTLLM numbers) and dfly acceptance.
4. The hook's masking takes effect inside the Triton modular kernel under CUDA graphs. The sidecar wait counter must
   stay 0 (`e3-stats-rank*.json` timeouts) and `cos_gate.py parse` must pass with `MIN_COS`.
5. End-to-end quality of A-mxfp4 vs B (GSM8K-200 or the ACS gates). The offline numbers say MXFP4 warm experts move
   per-layer outputs by about 2% cosine; whether that costs accuracy is only measurable on the model.
6. Real routing skew (COUNT run) and the rowmap's held-out cold share; sidecar prep time and 6000 memory peak
   (4,283 experts, 37 layers x 14 graph buckets). The sidecar keeps the converted source tensors (`keep.append(W)`,
   as on DS-V4, where b12x preparation aliased them). If b12x makes a full repacked copy of the converted Hy4 tensors,
   peak memory roughly doubles and 80 GiB of warm experts will not fit. Watch the sidecar's "peak reserved" line, and
   lower `--capacity-gib` in the rowmap if needed.
7. The hook installs in every vLLM process: `E3 patched ... mode=... geometry hy_v4 E=256 top-k=8` in both server logs.
   `hy4-up.sh` A/COUNT checks this and fails otherwise, because `hook/sitecustomize.py` swallows install errors and a
   broken hook would quietly serve B.
   (Checked in the pulled image: its only `sitecustomize` is Ubuntu's apport hook, `/etc/python3.12/sitecustomize.py`,
   which `hook/sitecustomize.py` reproduces. No PYTHONPATH is set.)

## First GPU smoke (exact commands)

Run these after the weights finish on both Stations. Everything below the staging step runs on station-left unless
it says otherwise.

```bash
# 0. stage (from the Threadripper)
for h in station-left station-right; do
  rsync -a --delete --exclude .git -e "ssh -i ~/.ssh/id_ed25519_station_ctl" \
    ./ exx@$h:~/hy4-e3/repo/
  ssh -i ~/.ssh/id_ed25519_station_ctl exx@$h 'docker pull vllm/vllm-openai:hy4 && \
    docker image inspect vllm/vllm-openai:hy4 --format "{{index .RepoDigests 0}}" && \
    ls /models/Hy4-FP8/model-*-of-00245.safetensors | wc -l && sha256sum /models/Hy4-FP8/config.json'
done   # expect 245 shards, config sha256 2ae591dd...

# 1. b12x one-layer check on the 6000 (minutes; RTX must be empty). station-left RTX:
SC=vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b
for f in mxfp4 mxfp8; do
docker run --rm --device nvidia.com/gpu=GPU-7d8548c5-b79e-964c-433b-e601a28a53f1 \
  -v /models/Hy4-FP8:/model:ro -v ~/hy4-e3/repo:/e3:ro -v ~/dsv4pro-e3/b12x:/b12x:ro \
  -e PYTHONPATH=/b12x -e E3_HOOK_DIR=/e3/hook -e E3_CONFIG=/model/config.json --entrypoint python3 $SC \
  /e3/tests/b12x_layer_check_hy4.py --model /model --layer 20 --experts 0-35 --wfmt $f
done   # PASS = cos(b12x, emu) >= 0.999 and cos(b12x, fp8 ref) ~0.98 (mxfp4) / ~0.997 (mxfp8)

# 2. Recipe B (baseline) and teardown
bash ~/hy4-e3/repo/launch/hy4/hy4-up.sh B ~/bench/hy4-B-$(date +%m%d-%H%M)
#    check: both server logs show "Total CPU offloaded parameters: 166.5" (not 169.5) and the TRITON FP8 MoE backend
curl -s localhost:8020/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"hy4","messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":256}'
bash ~/hy4-e3/repo/launch/hy4/bench-hy4.sh ~/bench/hy4-B-bench 3100 3 "1 16 32"
bash ~/hy4-e3/repo/launch/hy4/hy4-down.sh

# 3. COUNT run -> first real rowmap (traffic: the bench above, or real prompts)
L=~/bench/hy4-COUNT-$(date +%m%d-%H%M); bash ~/hy4-e3/repo/launch/hy4/hy4-up.sh COUNT $L
python3 ~/hy4-e3/repo/tools/cos_gate.py send http://127.0.0.1:8020 hy4 ~/r21/gsm8k_test.jsonl 6
bash ~/hy4-e3/repo/launch/hy4/bench-hy4.sh $L/bench 3200 1 "16"; sleep 10
python3 ~/hy4-e3/repo/tools/counts_delta.py $L/e3-counts-ready-rank0.json $L/e3-counts-rank0.json $L/delta-rank0.json
bash ~/hy4-e3/repo/launch/hy4/hy4-down.sh
#    rowmap (anywhere with numpy): build_rowmap.py --config models/hy4/config.json --counts delta-rank0.json \
#      --offload-gib 166.5 --warm-fmt mxfp4 --capacity-gib 80 --out rowmaps/rowmap-hy4-v1.json

# 4. Recipe A with the cosine gate (prior rowmap; sidecars first, one per Station)
E3ROWMAP=rowmap-hy4-prior-mxfp4.json WFMT=mxfp4 bash ~/hy4-e3/repo/launch/hy4/hy4-sidecar.sh 0
ssh exx@192.168.101.2 'E3ROWMAP=rowmap-hy4-prior-mxfp4.json WFMT=mxfp4 bash ~/hy4-e3/repo/launch/hy4/hy4-sidecar.sh 1'
L=~/bench/hy4-A-$(date +%m%d-%H%M)
E3ROWMAP=rowmap-hy4-prior-mxfp4.json E3_CHECK=4 E3_CHECK_MIN_T=1024 bash ~/hy4-e3/repo/launch/hy4/hy4-up.sh A $L
python3 ~/hy4-e3/repo/tools/cos_gate.py send http://127.0.0.1:8020 hy4 ~/r21/gsm8k_test.jsonl 6
scp exx@192.168.101.2:$L/server-rank1.log $L/server-rank1.remote.log
MIN_COS=0.97 python3 ~/hy4-e3/repo/tools/cos_gate.py parse $L/server-rank0.log $L/server-rank1.remote.log \
  ~/hy4-e3/repo/rowmaps/rowmap-hy4-prior-mxfp4.json
grep -h timeouts $L/e3-stats-rank0.json; bash ~/hy4-e3/repo/launch/hy4/bench-hy4.sh $L/bench 3100 3 "1 16 32"
bash ~/hy4-e3/repo/launch/hy4/hy4-down.sh sidecars
```

## Risks, ranked

1. **MXFP4 warm numerics.** The per-layer cosine against FP8 is about 0.98 (offline, real weights). It is the
   specified format, but A-mxfp4 vs B is then also a quantisation comparison, not only a placement comparison. A needs
   an accuracy gate next to throughput. MXFP8 (cos about 0.997) is the clean-numerics arm at about half the warm
   capacity.
2. **Backend.** `--moe-backend triton` is forced: the hook needs a modular, expert-map-honouring, UVA-preserving FP8
   backend. Triton block-FP8 on SM103 is untested here and likely slower than the vendor's auto TRTLLM. B and A use the
   same backend, so the comparison stays fair, but absolute numbers will not match a vendor TP8 recipe.
3. **First boot of the hy4 image on GB300 + SM103 with EP2 + UVA + dfly.** None of it has run. Specific risks: HBM
   margin (about 2 GiB estimated), FLASHMLA_SPARSE + TP2, dfly with `--headless` rank 1, and the iHC sequence-parallel
   path with EP.
4. **b12x contract for converted weights.** Layout "w31", scale clamp, swiglu clamp and MXFP8 alpha are checked
   only by emulation until `b12x_layer_check_hy4.py` runs. A wrong layout fails loudly (cos near 0); it is not silent.
5. **Rowmap quality.** The prior is uniform. With MXFP4 at 90% warm, the counts matter less; with MXFP8 (47%) they
   matter, so run COUNT before any MXFP8 A number is reported.
6. **Single Station** does not fit without an exclusive warm loader (not written), and even then the Grace margin
   is only 28.5 GiB.
7. **Verified against the pulled image (amd64, 2026-11-21).** Its `vllm/` Python tree is byte-identical to
   `thisjiang/vllm@5790aa8` (`diff -rq`, `.so` and `__pycache__` excluded; the image adds only `vllm-rs/`). The arm64
   image the Stations will pull is built from the same commit (config `VLLM_BUILD_COMMIT`), but its files were not
   diffed.