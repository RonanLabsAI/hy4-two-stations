# L6-4h: Hy4 (770B / 49B active) down a precision ladder, and an RTX warm-expert sidecar for FP8

**DRAFT, not frozen. Not executable.** Written 2026-11-20 PT for PI review. Every value marked **TBD** must be fixed
and every item in "Freeze checklist" done before this file is frozen, hashed and pushed to RonanLabsAI/preregistrations.
Times are PT (America/Los_Angeles). Lane L6 (approved 2026-11-17, card L6a); this study is an L6-4-style
capacity-vs-precision question on one model that needs no Sparks. Design: `docs/l6/DESIGN.md`. Predictions:
`tools/l6-model/hy4model.py`, `tools/l6-model/data/hy4-predictions.json`. Every outcome is recorded publicly,
including nulls, gate stops and unrun dispositions.

## Questions

- **L (ladder).** How much measured quality does Hy4 lose at each rung below FP8 (Q4_K_M ~4.89 bpw, IQ2 ~3.45 bpw,
  STQ1_0 ~2.39 bpw), and what speed does each rung give in the placement that fits our hardware? Quality is
  measured against FP8-B on identical items and identical token positions.
- **S (sidecar).** Does moving FP8-B's Grace-resident cold experts onto each Station's RTX PRO 6000 as MXFP4 warm
  experts ("A") earn at least **1.05x** throughput at C1 or C16 over B, while matching B's quality within the
  frozen tolerance below? (Warm experts change precision, so speed alone is not enough.)
- **D (drafter, descriptive).** What acceptance and ms per verify step does the built-in DFly2 drafter give on our
  FP8-B load at 2K, 16K, 128K and 512K context? The vendor claims a constant draft cost from index sharing.

## Model and source claims

- `tencent/Hy4-FP8` @ `d063b5853ecf56e0776266dab28305fc17a137cc` (245 shards, 786,294,748,086 bytes per
  `model.safetensors.index.json`). Licence: Tencent Hy License (Apache-2.0 terms plus a MaaS clause that does not
  apply to us; full text `tools/hy4/data/hy4-fp8-d063b585/LICENSE`, SHA-256 `52755a67...`).
  Architecture (config.json, model card): 78 layers (layer 0 dense, 77 MoE), hidden 6144, 256 routed experts + 1
  shared, top-8, expert intermediate 2048, vocab 120,832, MLA (q_lora 2048, kv_lora 512, rope 64) with gated DeepSeek
  Sparse Attention (indexer top-2048; 21 "full" indexer layers, 57 reuse an earlier index), iHC with 4 residual
  streams, 1M context, VL (ViT 1.23B), built-in 5-layer DFly2 drafter (3.3B, block size 7; vendor recipe uses
  `num_speculative_tokens` 5).
- Byte inventory, read from the safetensors headers by HTTP range (`tools/l6-model/data/hy4-fp8-tensor-bytes.json`):
  routed experts 744.29 GB FP8 (9.666 GB per MoE layer), attention 26.34 GB (o_proj and gate FP8, the rest BF16),
  shared experts 5.81 GB BF16, lm_head 1.48 GB, embeddings 1.48 GB, vision 2.47 GB, DFly2 3.37 GB. Per decoded token
  the backbone reads ~34.7 GB of non-routed weights plus ~23.3 GB of routed experts (1/32 of them) before batch reuse:
  **Hy4 is attention-heavy**, so bits spent on experts change C1 speed less than on GLM-5.3.
- GGUFs: `AngelSlim/Hy4-GGUF` @ `3ab6319e385d482075b5e9eb6e4bc1c3ffede8ec`, converted from the VL checkpoint.
  Per-tensor bytes from the GGUF headers (`tools/l6-model/data/hy4-gguf-tensor-bytes.json`; every file sum equals the
  published size):

| Rung | File (HF LFS SHA-256) | Bytes | Routed experts | Non-routed (incl. head) | Expert types |
|---|---|---:|---:|---:|---|
| STQ1_0 | `Hy4-STQ1_0.gguf` (`ae822c4c...a3cd63`) | 230.48 GB | 206.8 GB | 23.7 GB | STQ1_0 / IQ2_XXS gate+up, IQ3_XXS / IQ4_XS down |
| IQ2 | `Hy4-IQ2.gguf` (`75884fbc...f942d3`) | 332.40 GB | 313.6 GB | 18.8 GB | IQ2_XS gate+up, Q4_K / Q6_K down |
| Q4_K_M | `Hy4-Q4_K_M.gguF` (`e3903003...2dfa95`) | 470.23 GB | 449.3 GB | 21.0 GB | Q4_K, Q6_K down on 37 layers |

  The Q4_K_M file name ends in a capital F in the repo and on station-right; scripts use it verbatim.
- Vendor sampling: temperature 0.9, top_p 1.0. **Deviation, forced by paired design:** every scored request here
  is greedy (temperature 0), as in L6-2 and Study D. Thinking control: `chat_template_kwargs.reasoning_effort`
  in {`no_think`, `high`}, default `high`.
- The GGUF README states the STQ1_0 patch has no MMQ kernel (prefill on STQ1_0 tensors uses dequantise + cuBLAS),
  and that `--jinja` is needed for chat. The harness never relies on server chat templating (see Workloads).

## Hardware and fixed environment

- Two DGX Station GB300 ("left", "right"): per Station one GB300 (256,703 MiB visible), one RTX PRO 6000 Blackwell
  Max-Q (97,887 MiB, 1,792 GB/s), Grace with 494 GB visible LPDDR5X, 72 cores. Link: direct ConnectX-8 400G DACs,
  RoCE v2. GPUs pinned by UUID (private), read at launch with `nvidia-smi`.
- **The RTX PRO 6000s are in scope for this study only** (rungs IQ2, Q4_K_M and arm A). That is a change from L6-1
  to L6-4, which exclude them; the PI must confirm no daily or route uses either RTX during the windows (**TBD**).
- Driver **TBD** (595.91.07 as of 2026-10-07), kernel **TBD**, host CUDA 13.2.51 (read 2026-11-20).
- **llama.cpp:** `Little0o0/llama.cpp` @ `f8d4cc414` (head of PR #32517, Hy4 support) plus
  `0001-stq1_0-quant-and-cuda.patch` (from the GGUF repo, SHA-256 **TBD**), CUDA archs `103a;120a`.
  The trees built 2026-11-20 at `~/hy4/llama.cpp` on both Stations used **host nvcc 13.2.51** with
  `LLAMA_BUILD_TESTS=OFF` and `GGML_RPC=OFF`. Our Kimi K3 receipt (`station-recipes/kimi-k3-llamacpp-c2c`) found
  13.2 cicc silently miscompiles llama.cpp's IQ1_S / IQ2_S / IQ3_S paths on GB300 (dropped `& 0xff` byte mask; PPL ~300
  instead of 2.04). Our rungs use IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_XS, STQ1_0 and K-quants, not the three types
  named, but the defect class (packed-byte reads) may reach the new STQ1_0 CUDA kernels. So gate G0 below applies,
  and the frozen build is a **CUDA 13.0 build** at `~/l6-4h/llama.cpp-cu130` (same commit, patch and archs,
  `LLAMA_BUILD_TESTS=ON`), built inside the CUDA 13.0.88 container we used for K3 (image digest in that receipt).
- **vLLM:** image `vllm/vllm-openai:hy4` (the vendor's tag). **Not present on either Station as of 2026-11-20**; it
  must be pulled and pinned by digest (**TBD**) before stage 2. The sidecar arm needs our port (branch
  `feat/hy4-sidecar`, being written in parallel) built on that image (**TBD: commit and image digest**).
- Harness: `tools/hy4/hy4bench.py` (stdlib Python, SHA-256 **TBD at freeze**; draft `c728c0b1...`), analysis
  `tools/hy4/analyze.py` (draft `810607e3...`), inputs frozen by `tools/hy4/build_inputs.py` into
  `tools/hy4/data/frozen/` (MANIFEST SHA-256 draft `84b13512...`). Tests: `uvx pytest tools/hy4/tests` (31 pass
  against a mock of both servers, pass and gate-fail paths) and `tools/l6-model/tests` (30 pass).

## Rungs, placements and stages

| ID | Rung | Engine | Station | Placement (all from `hy4model.py`, exact flags in Appendix A) |
|---|---|---|---|---|
| S1 | STQ1_0 | llama.cpp | left | GB300 only: all layers, KV 163,840 tokens. Predicted free 10.9 GiB |
| S2 | IQ2 | llama.cpp | left | GB300 + RTX, all GPU: whole model on the GB300 except routed experts of layers 1-23 on the RTX (`-ot`). Free 4.1 / 2.8 GiB |
| S3 | Q4_K_M | llama.cpp | right | GB300 + RTX + Grace: experts of layers 1-23 in Grace computed on the CPU (`--n-cpu-moe 24`), layers 24-40 on the RTX, rest on the GB300. Free 3.4 / 1.7 GiB |
| S4 | Q4_K_M | llama.cpp RPC | both | optional: two GB300s, layers 39-77 on right. Needs a `GGML_RPC=ON` build; runs only if stage-1 budget remains |
| B | FP8 | vLLM | both | EP over two Stations (TP2 attention + expert parallel); per Station 56.7% of its experts in HBM, 43.3% in Grace (cold), DFly2 on |
| A | FP8 | vLLM + sidecar | both | B, plus each Station's cold experts as MXFP4 on its RTX (all of them fit, 11.9 GiB spare); Grace unused for experts |
| A1 | FP8 | vLLM + sidecar | one | stretch: one Station, 25.8% HBM / 24.9% RTX / 49.3% Grace; only if stage 3 finishes inside its cap |

S2b (IQ2 with whole layers 57-77 on the RTX, `-sm layer`) is modelled for comparison and is **not** run: the
model predicts it slower than S2 at C1 (43.7 vs 50.8 tok/s), because the RTX would then read attention weights
every token.

**Stage 1** (single-Station rungs, both Stations at once): left runs S1 then S2; right runs S3. **Stage 2**: B, the
quality and logprob reference, plus the DFly2 check. **Stage 3**: A vs B in A-B-B-A order (loads A1, B1, B2, A2),
with quality on both A loads. A1 (one Station) only after stage 3, if budget remains.

## Predictions (recorded before any run)

`tools/l6-model/data/hy4-predictions.json`, SHA-256
**`ae2ca1d6e53809346bcbb3d663743afb297741875da9c212f2e1d5f070742f4e`** (re-generate with
`python3 tools/l6-model/hy4model.py --write`; a test checks the file against the code). Resident context 4,096
tokens per user, greedy. Bands are sensitivity ranges, not confidence intervals: for llama.cpp rows an explicit
x0.5 / x1.6 judgment, because **no llama.cpp GB300 receipt with known bytes per token exists** (the profile is an
uncalibrated prior that stage 1 calibrates); for vLLM rows the L6 GLM-5.3 spread with the 0.70 factor for an
unmeasured layout (x0.300 / x1.883).

| ID | C1 tok/s | C16 aggregate tok/s | C16 per user | Prefill 4K / 32K (input tok/s) | C1 bottleneck |
|---|---:|---:|---:|---:|---|
| S1 STQ1_0 | 31.4 / **62.8** / 100.5 | 153 / **307** / 490 | 19.2 | 1,248 / 1,223 | GB300 HBM reads |
| S2 IQ2 | 25.4 / **50.8** / 81.3 | 75 / **150** / 240 | 9.4 | 1,248 / 1,223 | GB300 HBM reads |
| S3 Q4_K_M | 9.4 / **18.8** / 30.0 | 15.6 / **31.3** / 50.0 | 2.0 | 872 / 860 | Grace CPU experts |
| S4 Q4_K_M RPC | 26.7 / **53.4** / 85.4 | 88 / **176** / 282 | 11.0 | 1,248 / 1,223 | GB300 HBM reads |
| B | 15.8 / **52.7** / 99.3 | 41 / **136** / 257 | 8.5 | 2,713 / 3,730 | Grace cold-expert reads |
| A | 22.8 / **76.2** / 143.4 | 117 / **391** / 735 | 24.4 | 5,182 / 5,546 | fitted per-cycle overhead |
| A1 | 12.6 / **41.9** / 78.8 | 32 / **105** / 198 | 6.6 | 1,338 / 1,991 | Grace cold-expert reads |

- Predicted A/B: **1.44 at C1, 2.87 at C16** (central). The C16 gain comes from Grace reads of the batch's expert
  union, which the model charges serially, with no prefetch overlap; it is the least trusted number in the table.
  Sensitivities (C1 / C16, central): DFly2 acceptance 0.55 gives B 38.7 / 100, A 56.0 / 287; acceptance 0.80
  gives B 66.2 / 171, A 95.5 / 490; no drafter gives B 23.1 / 67.9; uniform routing (locality 0) gives B 38.2 / 81.7;
  a 4x wire width for iHC gives B 50.5 / 122.
- S3 with the K3 C2C zero-copy patch (GB300 reads Grace experts instead of the CPU computing them) would be
  23.6 tok/s at C1; the patch is not in this tree and is not an arm here.
- DFly2: predicted C1 ms per verify cycle for B is 55.8 / 55.8 / 55.9 / 56.5 at 2K / 16K / 128K / 512K, flat
  within 1.3%, because sparse attention reads at most 2,048 latent rows and the indexer scan is small. A measured
  slope several times larger means the verify or draft path does not use the sparse kernel at long context.
- The model's llama.cpp prefill is a placeholder prior (one compute efficiency for all rungs); prefill is measured,
  and its prediction is reported but carries no weight.

## Workloads

All prompts are sent as **frozen token IDs** to `/v1/completions` (llama-server and vLLM both accept an ID array).
Rendering and tokenization happen once, on the desk, with the checkpoint's own `chat_template.jinja` and
`tokenizer.json` (SHA-256 `11b12531...`, `2e3f3d22...`) via `build_inputs.py` (tokenizers 0.23.3, jinja2 3.1.6,
pyarrow 26.0.1; deterministic, re-run checked byte-identical). This removes BOS and template differences between
engines; every request is gated on `usage.prompt_tokens == len(ids)`.

- **S (speed).** Study C's 16 public prompts (`reference-corpus.json`, SHA-256 `edb21df9...`), rendered thinking
  off. 512 output tokens with `ignore_eos`, temperature 0, seed 1234. C1: 16 prompts one at a time, streamed only to
  time the first token; tokens from `usage`. C16: all 16 at once, non-streamed; aggregate = completion tokens /
  (last end - first start). One warm round plus 5 retained rounds per cell.
- **P (cold prefill).** 4,095- and 32,768-token prompts; 8 tokens after the system header are replaced per request
  by a fresh random draw from a 26-token nonce pool, so no two requests share more than the template header.
  3 reps per length; prefill = prompt_tokens / TTFT; median reported. Cold gate: `cached_tokens` (when reported)
  stays below header + 8.
- **N (needle).** Exact-match retrieval of 3 random 8-character codes at 25/50/75% depth in a filler document,
  3 documents at 32,512 tokens and 3 at 130,816 tokens, thinking off, at most 96 output tokens. A code counts if
  it appears token-bounded in the reply. Pass at a length = all 9 codes found.
- **Q1 (GSM8K, every load that is scored).** First 500 test problems (openai/gsm8k, MIT), the L6-2 Q1 choice, zero-shot
  with a fixed instruction ending `#### <number>`, thinking off, at most 1,024 tokens, 16 in flight. Study D scoring
  rule (number after `####`, else the last number; commas removed; numeric equality).
- **Q2 (MMLU-Pro, every load that is scored).** 500 items (TIGER-Lab/MMLU-Pro test, MIT), stratified by category
  (largest-remainder quotas) with seed 20261120, shuffled with the same seed. L6-2's subset seed is still TBD; if
  L6-2 freezes its draw first, this study adopts it and re-hashes. Zero-shot, thinking off, at most 1,024 tokens.
  Score: the last `answer is (X)`, else the last `(X)`, else wrong.
- **Q-think (small subset).** The first 40 items of Q1 and of Q2, `reasoning_effort: high`, cap 16,384 tokens
  (8,192 for S3, budget), 8 in flight (KV budget). Only the text after the last `</think:opensource>` is scored;
  cap hits and empty answers are counted. Descriptive: 40 items cannot resolve rung differences (see MDD).
- **LP (logprob drift).** 24 windows of wikitext-2-raw test text (CC BY-SA 3.0) and 24 of HumanEval prompt +
  canonical solution (MIT), 1,024 tokens each, no chat template. At every 16th position from 64 (60 per window, 2,880
  in all) the harness sends `ids[:p]` and records the top-64 next-token logprobs of a 1-token greedy completion;
  the true next token `ids[p]` is stored for NLL.

### Logprob modes, verified from source on 2026-11-20

- **llama-server** (the patched tree, read on station-left): `/v1/completions` rejects `echo`; there are no prompt
  logprobs. `/completion` with `n_probs` returns generated-token probabilities. With `post_sampling_probs` false
  (default) they are a softmax of the raw logits over the full vocabulary (`get_token_probabilities`), before
  temperature or top-p. Token-ID prompts are accepted, `cache_prompt` reuses the slot's prefix, and
  `tokens_evaluated` equals the full prompt length.
- **vLLM** (v0.31.0 source; the `hy4` image's version is **TBD**): `/v1/completions` supports generated-token
  `logprobs` (capped by server `--max-logprobs`, default 20, so **`--max-logprobs 64` is required**) and
  `prompt_logprobs` (non-streaming). `logprobs_mode` defaults to `raw_logprobs`; `return_tokens_as_token_ids`
  returns `token_id:N` keys.
- **Design:** the only mode both engines share is the generated-token logprob of a 1-token completion, so every rung
  is probed that way at the same positions with the same IDs. vLLM's `prompt_logprobs` on the B reference is an
  optional cross-check of that procedure, not a measure (**TBD**: whether the `hy4` image allows prompt logprobs
  with DFly2 speculation on).
- **Tokenizer alignment:** the GGUFs carry their own vocabulary. `check` stops the run unless the server
  detokenizes a frozen probe's HF IDs to the same text and maps each of the 6 special tokens in our prompts to the
  same single ID. Whether the server's own text tokenization matches HF is reported, not gated, because the
  harness never sends text.

## Thinking control (Study D amendment D-1 lesson)

The template renders `reasoning_effort` into the system turn and, for `no_think`, ends the prompt with an empty
`<think:opensource></think:opensource>` block; `high` (the default) ends with an open `<think:opensource>`. Both
strings are frozen in `templates.json`. Gates per request: prompt tokens equal the frozen count; for thinking off,
no reply contains a think tag; for thinking on, the reply closes the think block or hits the cap.

## Measures

- **Quality (per rung and arm):** Q1 and Q2 accuracy with Wilson 95% intervals; paired differences against B on
  the same items (Newcombe method-10 interval, exact McNemar); Q-think accuracy, cap hits and empty answers; needle
  hits per document.
- **Drift (per rung and arm vs B):** mean bucketed KL(B || rung) over the 2,880 positions, with a cluster bootstrap
  over windows (10,000 resamples, seed 42) for the 95% interval; median and p99 KL; top-1 agreement; mean change in
  true-token NLL; the share of positions where a true token falls outside a top-64 list (floored, flagged). Text
  and code are reported separately. The estimator is defined in `analyze.py`: both distributions are
  coarse-grained to B's top-64 IDs plus one tail bucket, and a B token missing from the test list gets the largest
  probability the test list allows. So KL can only be underestimated. **Noise floor:** KL(B1 || B2) between the
  two B loads of stage 3.
- **Speed:** S throughput per cell (5 retained rounds, mean and CV), C1 mean decode tok/s per reply, P prefill,
  load time to health, GPU clocks, power, temperature and memory every 2 s with no CUDA context, per-device
  buffer sizes and KV size from the server log (these replace the model's estimates).
- **Drafter (B, and A when on):** acceptance rate and mean accepted length from `/metrics` deltas (counter names
  **TBD** from the `hy4` image), separately for S and N; ms per verify step at C1 = decode wall / verify steps, at
  four prompt lengths (2K, 16K, 128K, 512K; thinking off, 256 output tokens, 3 reps). Ms per step is regressed on
  context length.
- **Predicted vs measured** for every row of the prediction table, with the model's band.

## Decision rules

### Minimum detectable differences (80% power, two-sided alpha 0.05, paired)

| Measure | n | MDD | Assumption |
|---|---:|---:|---|
| Q1 GSM8K-500, rung - B | 500 | **2.8 pts** if 5% of items are discordant; 4.0 pts at 10% | 2.80 x sqrt(discordant share / n) |
| Q2 MMLU-Pro-500, rung - B | 500 | **4.9 pts** at 15% discordant; 6.3 pts at 25% | same |
| Q-think subsets | 40 each | ~14 pts | descriptive only |
| Needle | 3 docs x 3 codes per length | none | pass / fail, descriptive |
| Drift, mean KL | 2,880 positions, 48 clusters | 2.80 x bootstrap SE of the mean | reported with the result; the SE is not known before B runs |
| A vs B speed | 10 pairs per cell | ~2.0% at 2% SD of the log ratio | (2.262 + 0.883) x SD / sqrt(10) |

The realised discordant share and MDD are reported with every quality comparison, so a null comes with its
power. "No measurable loss" means the 95% interval includes zero; it is not a claim of equivalence. Equivalence
is claimed only where an interval lies inside a frozen tolerance.

### L, per rung (against B)

- **Quality loss** at a rung if the Q1 or Q2 paired interval (rung - B) lies entirely below zero; **no measurable
  loss** if both intervals include zero; **equivalent** if both lie inside ±3 points (Q1) and ±5 points (Q2).
- **Drift** at a rung is "above noise" if the lower 95% bound of mean KL(B || rung) exceeds the upper 95% bound of
  KL(B1 || B2). Reported for all, text and code.
- The ladder outcome is the table of these verdicts plus the speeds. No rung is "admitted" or recommended by
  this study; picking a serving rung is the PI's call.
- A rung whose check, prompt or thinking gate fails is "not comparable" for the affected measure and is reported
  as such.

### Speed, per rung

The cell value is the mean of 5 retained rounds; interval Student t95, df 4 (critical 2.776). It is reported against the
prediction band: "inside" or "outside, high / low". Stage 1 recalibrates the llama.cpp profile (memory efficiency
and per-step overhead from S1's C1 and C16) and records the refit before stage 2 runs. Rungs are not compared
with a significance test: each has its own placement, so differences are descriptive.

### S, the sidecar (A vs B)

- **Order:** A-B-B-A over four loads (A1, B1, B2, A2). Each load: one warm plus 5 retained S rounds at C1 and C16, P
  once, LP once; Q1 and Q2 on A1 and A2 (B's quality is stage 2's, and B1 also re-runs Q1 as a check of
  load-to-load stability). 10 pairs per cell: round i of A1 with round i of B1, and round i of B2 with round i of
  A2.
- **Statistic:** mean paired log ratio A/B, Student t95, df 9 (critical 2.262157), endpoints back-transformed.
- **A is faster** if the t95 lower bound exceeds **1.05** at C1 or at C16.
- **Quality tolerance (frozen; all must hold, on the pooled A loads vs stage-2 B):**
  1. Q1 paired difference (A - B), Newcombe 95% interval inside **[-3, +3] points**;
  2. Q2 paired difference, 95% interval inside **[-5, +5] points**;
  3. drift: the upper 95% bound of mean KL(B || A) is at most **mean KL(B1 || B2) + 0.010 nats**, and top-1
     agreement(B, A) has a lower 95% bound at least top-1(B1, B2) - 0.02;
  4. needle: A finds every code that B found, at both lengths.
- **Verdicts:** "A admitted" = faster and all four tolerance tests pass. "Faster, not shown equivalent" = faster, with any of
  tests 1-3 straddling its bound. "Faster, worse" = faster, with an interval lying wholly outside a bound or test 4
  failing. "Not faster" otherwise. The PI decides any recipe change.
- **B stability:** B1 and B2 S rounds must each lie within ±3% of their pooled mean at each cell, with CV at most 2%;
  otherwise the A-B-B-A block is void for that cell and is reported as such.
- **K- (sensitivity control):** before stage 3, one B load with `NCCL_PROTO=Simple` (L6-2's candidate; costs at
  least 10% at decode sizes in our October tuner work) runs 3 S rounds at C16. Its ratio to B must be below 0.95,
  or the harness is declared insensitive and stage 3 halts. **TBD:** confirm that `model.md` predicts at least a 10%
  cost for it on Hy4's message sizes.

### D, drafter

Descriptive. Reported: acceptance per context, ms per verify step, the regression slope with its t95 interval,
and the predicted slope. A slope above 3x the predicted one is flagged as "draft or verify path not sparse at long
context" and becomes a candidate for a later card.

## Budget (GB300-hours: each GB300 per wall hour; RTX hours reported separately)

Estimates use the central predictions; llama.cpp rows double them because their band is wide.

| Block | Wall (est.) | GB300-h |
|---|---|---:|
| G0 build and gate (CUDA 13.0 build, test-backend-ops, canary, PPL) on both, no model load beyond a canary | 1.0 h | 2 |
| Stage 1 left: S1 STQ1_0 (~1.3 h central) then S2 IQ2 (~1.8 h), with restores | 4.0 h cap | 4 |
| Stage 1 right: S3 Q4_K_M (~6 h central; thinking subset capped at 8,192 tokens) | 7.0 h cap | 7 |
| Stage 2: B load (~0.75 h), all workloads, DFly2 check, restore | 4.5 h cap | 9 |
| Stage 3: K- (0.5 h), A-B-B-A (4 loads, ~1.5 h each) | 6.5 h cap | 13 |
| A1 stretch (one Station) | 2.0 h cap | 2 |
| S4 optional (two Stations, RPC build) | 2.0 h cap | 4 |
| **Total cap** | | **41** |

Budget rule (as L6-2): before any load, if elapsed + that load's estimate (measured round times once available) +
a 0.5 h restore reserve exceeds the window or the block cap, the load does not start and the remaining blocks are
recorded as unrun (budget). Inside a load, a workload that would cross the cap is cut at a whole-item boundary and
the cut is recorded. S3's 6 h estimate makes the right Station the critical path of stage 1. If the PI wants
stage 2 sooner, the pre-registered alternative is to drop S3's Q-think subset (saves ~2.5 h). Choose at freeze
(**TBD**).

## Stops

- **G0 fails:** if `test-backend-ops -o MUL_MAT,MUL_MAT_ID` fails for any type used by the rungs (Q4_K, Q5_K, Q6_K,
  Q8_0, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_XS, STQ1_0, BF16, F16) in the CUDA 13.0 build, or the canary prompt or a short
  wikitext-2 PPL disagrees between the 13.0 build and the 13.2 build by more than 1% on STQ1_0, the affected rung
  is unrun and the defect is reported, upstream too. (The 13.2 comparison documents whether the existing
  `~/hy4` builds were safe; they are never used for measurement.)
- A GGUF or FP8 shard fails its SHA-256 check: no load.
- Any `check` gate fails (prompt tokens, detokenization, specials): that load stops; nothing is measured.
- A rung OOMs on load: the pre-registered fallback ladder in Appendix A (at most 2 steps, each recorded); a third
  failure makes that rung unrun.
- Thinking or D-1 gate failures above 2% of a workload's requests: that workload is "not comparable" for the load.
- B's S CV above 2% at any cell in stage 2: stage 3 does not start.
- K- not below 0.95: stage 3 halts as insensitive. (K- was replaced after it failed; see pre-freeze change 7.)
- NCCL INIT log shows socket transport instead of IB in stage 2 or 3: that load is void; twice in a row stops.
- Production needs either Station, the claim is revoked, a fallback route fails, or anyone changes a router or
  daily: stop and restore.
- Any step needing sudo that gets "password required": stop.
- A second consecutive failure of the same step: stop.

## Operations and interruption contract

- **Gate:** stage 1 needs a PI-approved window per Station. Stages 2 and 3 need both Stations and both dailies
  stopped (L6b rules or a PI-approved planned window). The RTX PRO 6000s are used only inside such a window, with
  the PI's confirmation that nothing routes to them.
- **Before the claim:** the downloads are finished: no `hy4-dl-*` container running. We never touch those
  containers or `~/hy4`. Every file's SHA-256 matches HF LFS. Fallback routes answer through Sparks. Snapshot the
  dailies' containers and the router identity. Production LiteLLM is never restarted or edited.
- Take a preemptible fleet claim covering the GB300(s) and RTX(es) used, by GPU UUID.
- Servers bind to loopback (`127.0.0.1:18080` llama-server; vLLM **TBD** port); NCCL binds to the direct-DAC
  addresses. The harness runs on the same Station, against loopback.
- **Restore** on any stop, error or cap: stop our servers and samplers, restart the preserved dailies, time each
  to health and to a routed answer, compare identities, release the claim. Restore time is recorded.

## Public outcome

`outcomes/l6-4h-hy4/` in RonanLabsAI/preregistrations: this protocol's SHA-256 and immutable commit URL; per rung
and arm the S round values, P, N hits, Q1/Q2/Q-think per-item correctness with item IDs (public sets only), drift
summaries and per-position KL (positions and token IDs, not text), drafter acceptance and step times, buffer and
KV sizes, gates, dispositions, GB300-hours and RTX-hours, restore times, the predictions file and predicted vs
measured, and the G0 result. Launch scripts are ours (written from scratch). Private hosts, IPs, interface names,
paths, device UUIDs and serials stay out. A post covers the result only if it is meaningful. A measured precision
ladder with paired quality on one 770B model is meaningful by default; so is a G0 compiler defect. The PI picks
the tier.

## Freeze checklist (all must be done before freezing)

1. PI windows: stage 1 per Station; stages 2-3 under L6b or a planned window; RTX use confirmed free of routes.
2. Downloads complete on both Stations; every GGUF and every FP8 shard checked against HF LFS SHA-256 (GGUF
   values in this file; FP8 manifest hash **TBD**). Note: as of 2026-11-20 each Station's HF cache holds a stale
   duplicate `.incomplete` partial for some files (24-40 GB each); the owner of the download job should clear them
   after it finishes. This study does not touch them.
3. G0: CUDA 13.0 build at `~/l6-4h/llama.cpp-cu130` on both Stations; `test-backend-ops` for the listed types; canary
   and PPL comparison against the 13.2 build; results hashed. Patch SHA-256 recorded.
4. `vllm/vllm-openai:hy4` pulled on both Stations and pinned by digest; its vLLM version recorded; `--max-logprobs 64`
   accepted; spec-decode metric names read from `/metrics`; whether `prompt_logprobs` works with DFly2 on.
5. Sidecar port (`feat/hy4-sidecar`) commit and image digest frozen; its warm-expert MXFP4 conversion documented
   (scales, which tensors), and a 20-prompt greedy smoke against B passes before any timing.
6. Stage 2/3 vLLM launch commands written out verbatim (Appendix B draft) and hashed.
7. `build_inputs.py` re-run, MANIFEST hash matches; L6-2 MMLU-Pro seed reconciled.
8. `hy4bench.py` and `analyze.py` hashed; `uvx pytest tools/hy4/tests tools/l6-model/tests` passes on the frozen
   commit; the harness copied to `~/l6-4h/tools/hy4` on both Stations and hashed there.
9. Predictions file re-generated from the frozen model (SHA-256 above unless the model changed; any change is
   recorded with its reason before the first run).
10. Quality tolerance values (±3 / ±5 points, KL + 0.010 nats, top-1 - 0.02) confirmed by the PI.
11. S3 Q-think decision (keep or drop) and S4 (run or unrun) decided.
12. Smoke inside an approved window: S1 loads, `check` passes, one S prompt and one LP window run, restore
    (counts toward stage-1 budget).

## Pre-freeze changes from the first stage-1 attempt (2026-11-20, 16:46-16:53 PT, nothing measured)

The first attempt stopped before any measurement on both Stations. Three findings changed the draft; all are
recorded here and in the run logs (`~/l6-4h/g0-attempt1/`, `~/l6-4h/runs/*-1646.log`, `*-1650.log`).

1. **G0 passed on both Stations** (CUDA 13.0.88 build, NCCL off, host-path RUNPATH). `test-backend-ops`
   MUL_MAT 1294/1294 and MUL_MAT_ID 397/397 OK on GB300 and RTX for every listed type including STQ1_0.
   STQ1_0 wikitext-2 PPL (c 512, 64 chunks) is 4.9152 +/- 0.1006 on both the 13.0 and the 13.2 build (relative
   difference 0.0), and the canary is byte-identical. The existing 13.2 builds were therefore safe for this
   model; measurement still uses the 13.0 build. A first 13.0 build was rejected by G0's link check (RUNPATH to
   the container path, NCCL linked); it was rebuilt with NCCL off to match the 13.2 build (all rungs use
   `-sm layer`, which does not use NCCL).
2. **One GPU per CUDA process.** Any process that sees both the GB300 and the RTX fails `ggml_cuda_init`
   ("initialization error"), as in L5 Finding 1. The IQ2 and Q4_K_M placements therefore run the RTX in its own
   `ggml-rpc-server` (127.0.0.1:50053) and `llama-server` on the GB300 with `--rpc 127.0.0.1:50053 -dev
   CUDA0,RPC0` and `-ot ...=RPC0[127.0.0.1:50053]` for the same expert tensors listed in Appendix A. The tensor
   placement is unchanged; each token adds a loopback RPC round trip per RTX-resident expert layer (23 on IQ2,
   17 on Q4_K_M), which the speed model does not include. The predictions stay as recorded; the comparison
   will note this.
3. **GGUF special-token names differ.** The AngelSlim GGUFs carry the same special IDs (120000, 120001, 120025,
   120029, 120030, 120039) under hashed names (`<think:6124c78e>`, `<｜hy_end:6124c78e｜>`), so the `check` gate's
   string-to-ID specials test failed on STQ1_0 although the probe detokenized and tokenized exactly. Changes:
   the specials gate is now an ID round trip (each special ID's server string tokenizes back to that one ID; the
   strings are recorded); think tags are matched as `</?think:[^>]*>`; every special-token string is stripped
   before scoring (hash digits would otherwise be read as a GSM8K answer). Also, llama-server only emits special
   tokens in completion text with `-sp`, and vLLM only with `skip_special_tokens=false`; without these the
   thinking gates and answer split would have silently failed. All servers now get `-sp`; vLLM requests send
   `skip_special_tokens=false, spaces_between_special_tokens=false`. Harness tests: 35 pass (4 new).
4. **Logprob probes whose greedy token is a partial UTF-8 piece** (found in the measured STQ1_0 run, 20:03 PT,
   probe 2761 of 2880 = `lp-code-22` position 64; Q4_K_M failed at the same probe). The greedy token there
   (ID 96859) is the first byte-piece of ' ➞'. llama-server stores a token and its probabilities only once the
   generated text is complete UTF-8 (`add_token` sits inside `if (!incomplete)` in server-context.cpp), streamed
   or not, so it returns no `completion_probabilities` and the harness crashed. Reproduced on a temporary Q4_K_M
   server: `n_predict: 2` returns only the second piece's probabilities, and `ignore_eos` does not help. No
   request-level workaround exists. Change: such a probe is recorded with an empty top-k, the sampled ID from
   `tokens`, `gate=false` and `no_probs: incomplete_utf8`; the summary counts them; `analyze.drift` already
   excludes any position that failed on either side of a comparison and reports the count. Positions excluded
   this way depend on each llama.cpp rung's own greedy token (vLLM is unaffected), so the count is reported per
   comparison. The harness also retries once with `ignore_eos` before giving up (covers a true EOG greedy
   token). STQ1_0, IQ2 and Q4_K_M logprob captures are re-run in full with this harness; their other workloads
   stand. Harness tests: 37 pass.

5. **Stage 2 B placement and the drafter workload** (2026-11-21 ~00:40 PT, before any stage-2 load). B runs from
   `feat/hy4-sidecar` `launch/hy4/hy4-up.sh B` with `CTX=163840` (the GGUF rungs' context, so every harness
   workload runs unchanged on B), `SEQS=16`, `--max-logprobs 64 --enable-prompt-tokens-details`, Triton MoE,
   `--cpu-offload-gb 166.5` (MoE layers 1-37 per rank on Grace). If the KV pool does not fit, the offload steps one
   MoE layer at a time (171.0, then 175.5; max 2 steps, recorded), as stage 1's ladder. The API binds 127.0.0.1
   (the branch had 0.0.0.0; fixed after review). The D workload was not in the harness: added `hy4bench drafter`
   (C1, thinking off, 256 output tokens with ignore_eos, 3 reps; spec-decode counter deltas around each request;
   ms/step = decode wall after the first token / verify steps) and `data/frozen/drafter.json` (wikitext-2 prose
   tiled to 2K/16K/128K/512K minus 256 tokens, plus a summarize instruction; SHA-256 663f01aa...; every other
   frozen file is byte-identical). B runs D at 2K/16K/128K. 512K does not fit next to B's placement at 163,840
   context, so it is a separate short load (D512: same flags, larger offload and `CTX=532480`, drafter only) sized
   from B's measured KV pool; its ms/step is compared with B's only as a trend, since the placement differs.
   Harness tests: 39 pass.

6. **Stage 3 setup** (2026-11-21 04:36 PT, before any stage-3 load). The A rowmap is built from a COUNT load
   (B placement plus routing-count capture) under frozen harness traffic (1 speed round, 100 quality items):
   `rowmap-hy4-v1-mxfp4.json` (SHA-256 d594dbdb...), 4,283 warm experts per rank (80 GiB), in-sample 97.8% of
   Grace-layer routes warm. Caveat: the counts come from the same frozen speed prompts that A is timed on, so the
   coverage is optimistic; at MXFP4 the warm set is 90% of all Grace-layer experts, so the effect is small. The
   sidecar kernel was checked on both RTXs before use (b12x vs emulation 0.99991; vs the FP8 checkpoint 0.978-0.986
   MXFP4, 0.997-0.998 MXFP8, layers 1/20/40/60/77; the swapped-layout control now fails at 0.54). Sidecar review
   fixes applied before use: the A1 gate also requires the identity cos(masked + sidecar, full) >= MIN_COS and that
   masking happened (prod_frac < 0.9999); a run with any sidecar timeout is invalid; each rank must mask at least
   the rowmap's warm layers. MIN_COS 0.97 (MXFP4). Budget: B1 does not re-run Q1 (an optional stability check),
   and sidecars are stopped during B1 and B2 (an idle sidecar spin-polls on Grace). K- runs speed at both cells
   (the harness has no single-cell option); only its C16 ratio to stage-2 B (93.22 tok/s) is gated.
7. **K- replaced after it failed** (2026-11-21 07:40 PT, PI decision; this change was made AFTER seeing a result).
   The first K- (`NCCL_PROTO=Simple`, run 04:36 PT, `runs/s3-kminus-20261121-043626`) gave C16 95.17 tok/s, ratio
   1.021, and C1 27.17 (ratio 0.984), so stage 3 halted as specified. The logs cannot show that NCCL applied the
   protocol (NCCL 2.31.2 with the tuner plugin loaded logs no protocol choice), and the TBD above (that `model.md`
   predicts at least a 10% cost on Hy4) was never confirmed, so the control was not shown to be a real slowdown.
   That first K- result is reported as is. The replacement K- is B with the drafter off (`SPEC=0`), everything else
   unchanged; it runs speed at both cells and gates on **C1**: its ratio to stage-2 B C1 (27.61 tok/s) must be
   below 0.95, or stage 3 halts. The speed model predicts a large cost (one token per step instead of ~2.9 at the
   measured 0.375 acceptance), so this control shows the pipeline responds to a known change in the timed path; it
   does not show sensitivity at the 5% scale. Fine-scale sensitivity rests on the B1/B2 stability rule (stage-2 B
   CV 0.1%). The C16 ratio is reported, not gated.

8. **Stage 3 result and two follow-ups** (2026-11-21 12:45 PT, written after stage 3 and before either follow-up
   runs; PI approved running them). Stage 3 (MXFP4 sidecar) gave "Faster, worse" under test 3: B1, B2 and stage-2 B
   logprobs are byte-identical, so the drift bound became KL <= 0.010 and top-1 >= 0.98, which any 4-bit expert
   conversion misses. That verdict stands as reported. For the follow-ups only, verdicts are given on two tiers:
   - **Lossless:** tests 1-4 exactly as above (floor = B1/B2 = 0).
   - **Near-lossless:** tests 1, 2 and 4 as above, plus: the upper 95% bound of delta NLL(A - B) is at most
     +0.020 nats; the upper 95% bound of mean KL(B || A) is at most 0.146 nats; the lower 95% bound of top-1
     agreement is at least 0.915. The last two are Q4_K_M's stage-1 point values: "drifts no more than Q4_K_M".
     These bounds were chosen after seeing stage-3 drift, which is why they apply only to runs made after this entry.
   Each tier is applied per A load; a tier passes only if both A loads pass it.
   **Follow-up order (one script, `stage3b-run.sh`):** A4 (MXFP4, v1 rowmap) | A8-1 (MXFP8, rowmap
   `rowmap-hy4-v1-mxfp8.json`, 2,206 warm experts per rank, cosine + identity gate at MIN_COS 0.995) | B3 | B4 |
   A8-2 (MXFP8) | A4b (MXFP4). Two questions:
   1. **Held-out traffic (MXFP4).** A new frozen set `speed_heldout.json` (SHA-256 ee846fb9...): 16 prompts in the
      speed shape, 8 encyclopedia requests on wikitext-2 article titles and 8 HumanEval completions (tasks
      120-127), none seen by COUNT. A4, B3, B4 and A4b each run it (1 warm + 5 rounds, C1 and C16); pairs A4-B3
      and B4-A4b, t95 df 9. "Faster on held-out" = lower bound above 1.05. Reported alongside: the held-out ratio
      divided by the stage-3 frozen ratio. A4 and A4b run held-out speed only.
   2. **MXFP8 sidecar.** A8-1, B3, B4, A8-2 form an A-B-B-A block under the stage-3 rules (frozen speed, prefill,
      logprobs, Q1, Q2 and needles on A8; frozen speed, held-out speed and prefill on B). B logprobs are not
      rerun: stage-2 B is the reference, since three B loads gave identical logprobs. The A8 loads also run
      held-out speed. Verdicts on both tiers.
   B stability rule as above (B3, B4). Any sidecar timeout voids that run.

## Appendix A: stage 1 commands (exact, not run)

Scripts: `tools/hy4/stage1/common.sh`, `left-stq1_0.sh`, `left-iq2.sh`, `right-q4_k_m.sh`. Each refuses to start
unless `L64H_APPROVED=1` is set and stops if a `hy4-dl-*` container is running, a used GPU has a compute
process, or the GGUF's SHA-256 differs. `CUDA_VISIBLE_DEVICES` is set by UUID in the order GB300, RTX, so
CUDA0 = GB300 and CUDA1 = RTX. Every server gets:

```
llama-server -m <GGUF> --host 127.0.0.1 --port 18080 --metrics --jinja \
  -fit off -fa on -lm none -np 16 -kvu -c 163840 -b 4096 -ub 2048 -t 64 -tb 72
```

`-fit off`: no automatic changes to our placement. `-c 163840 -kvu`: one unified KV of 163,840 tokens shared by
16 slots, enough for the 130,816-token needle plus output. `-lm none`: read weights into buffers, no mmap.

| Station | Rung | Placement flags |
|---|---|---|
| left | STQ1_0 | `CUDA_VISIBLE_DEVICES=<GB300>` `-ngl 999 -sm none -mg 0` |
| left | IQ2 | `CUDA_VISIBLE_DEVICES=<GB300>,<RTX>` `-ngl 999 -sm layer -ts 1,0 -mg 0 -ot 'blk\.([1-9]\|1[0-9]\|2[0-3])\.ffn_(gate\|up\|down)_exps\.weight=CUDA1'` |
| right | Q4_K_M | `CUDA_VISIBLE_DEVICES=<GB300>,<RTX>` `-ngl 999 -sm layer -ts 1,0 -mg 0 --n-cpu-moe 24 -ot 'blk\.(2[4-9]\|3[0-9]\|40)\.ffn_(gate\|up\|down)_exps\.weight=CUDA1'` |

Then on the same Station:

```
python3 -I ~/l6-4h/tools/hy4/hy4bench.py all --base http://127.0.0.1:18080 --engine llamacpp \
  --rung <rung> --out ~/l6-4h/runs/<rung>-<station>-<stamp>/bench --think   [--think-cap 8192 on Q4_K_M]
```

**Fallback ladder (max 2 steps, each recorded):** OOM on CUDA0 moves the next GB300 expert layer down one tier
(IQ2: layer 24 to the RTX; Q4_K_M: layer 41 to the RTX). OOM on CUDA1 moves its highest layer down one tier
(IQ2: back to CUDA0 with `-c 147456`; Q4_K_M: to Grace, `--n-cpu-moe` +1). The predicted free memory (GB300 3.4
to 10.9 GiB, RTX 1.7 to 2.8 GiB) is an estimate with compute buffers unmeasured, so a fallback is plausible on
IQ2 and Q4_K_M.

## Appendix B: stage 2-3 vLLM commands (draft, **TBD** at freeze)

From the vendor recipe, adapted to two Stations: `vllm/vllm-openai:hy4@<digest>` `vllm serve tencent/Hy4-FP8
--revision d063b585... --nnodes 2 --node-rank <r> --tensor-parallel-size 2 --enable-expert-parallel
--speculative-config '{"method":"dfly","num_speculative_tokens":5}' --attention-backend FLASHMLA_SPARSE
--reasoning-parser hy_v4 --tool-call-parser hy_v4 --limit-mm-per-prompt '{"image":0,"video":0}'
--max-model-len 532480 --max-num-seqs 16 --max-logprobs 64 --enable-prompt-tokens-details --host 127.0.0.1`,
plus our Grace cold-expert offload (B) or sidecar (A) flags from `feat/hy4-sidecar`, the NCCL environment from
L6-2's recipe A (one rail; host-specific HCA and GID), and `VLLM_USE_RUST_FRONTEND=1` as the vendor sets it. The
vendor recipe uses `--dtype bfloat16` and TP8 on one 8-GPU node; whether `--max-model-len` 532,480 fits next to
the hot experts is decided by the placement (**TBD**: KV pool size vs hot-expert share, which changes the B and A
predictions).
