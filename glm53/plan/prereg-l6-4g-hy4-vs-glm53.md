# L6-4g: Hy4 against full GLM-5.3 on two GB300 Stations, same harness, same items

**DRAFT, not frozen. Not executable.** Written 2026-11-22 PT for PI review. Every value marked **TBD** must be fixed
and every item in "Freeze checklist" done before this file is frozen, hashed and pushed to RonanLabsAI/preregistrations.
Times are PT (America/Los_Angeles). Lane L6 (approved 2026-11-17, card L6a). This study reuses L6-4h's harness,
frozen items and Hy4 results (`docs/l6/prereg-l6-4h-hy4.md`, outcome in RonanLabsAI/hy4-two-stations). It does not
replace L6-2 (NVIDIA's NVFP4 recipe, still unrun); it shares no arms with it. Every outcome is recorded publicly,
including nulls, gate stops and unrun dispositions.

## Why

The only published head-to-heads are Tencent's own blind rating (Hy4 won 46.8% of comparisons against GLM-5.3, lost
40.4%, tied 12.8%; 203 tasks, set not published) and Artificial Analysis's index (Hy4 35, GLM-5.3 32, mixed by
test). Our own GLM-5.3 two-Station numbers were measured on other prompts (SGLang NVFP4 84.8 / 588.0 tok/s at C1 /
C16, catid's prompt shape; TensorFold EXL3 73.2 tok/s C1, GSM8K 49/50) and carry no quality comparable with Hy4's.
No one has published both models on the same local hardware with paired quality.

## Questions

- **Q (quality, primary).** At released precision (FP8, both), does GLM-5.3 score differently from Hy4 on the same
  500 MMLU-Pro items and 500 GSM8K items, paired by item, with the same instruction, output cap and thinking off?
- **V (speed).** What decode speed does each model give in its best measured two-Station setup on the same frozen
  prompts, and in the matched FP8 setup (same engine, same layout rule)?
- **N (needles).** Does either model miss codes at 32K or 128K that the other finds?
- **T (thinking on, Part 2 only).** The same question as Q with thinking on, on a smaller subset, because that is
  how both vendors report and how the models are used.

## Models and arms

| ID | Model and precision | Engine | Layout | Source of data |
|---|---|---|---|---|
| H-B | Hy4 FP8 (`tencent/Hy4-FP8` @ `d063b585`) | vLLM `hy4` image | L6-4h arm B: TP2 + EP2, 166.5 GiB of experts per Station in Grace, DFly2 on | **existing** L6-4h stage 2 and stage 3 (B1, B2) |
| H-A | Hy4 FP8 + 4-bit (MXFP4) warm experts on each RTX PRO 6000 | vLLM `hy4` + sidecar | L6-4h arm A (4,283 experts per Station on the RTX) | **existing** L6-4h stage 3 (A1, A2) and held-out follow-up |
| G-B | GLM-5.3 FP8 (`zai-org/GLM-5.3` @ `ba32f8c1`, 141 shards, 755.6 GB, block FP8 e4m3 128x128) | the same vLLM `hy4` image | the L6-4h B rule applied to GLM: TP2 + EP2, KV for 16 x 163,840 tokens, 33 MoE layers of experts per Station in Grace (148.5 GiB); MTP drafter, 4 draft tokens | **new** |
| G-N | GLM-5.3 NVFP4 (`RadixArk/GLM-5.3-NVFP4` @ `6e389189`, 49 files, 464.9 GB) | SGLang `lmsysorg/sglang@sha256:352f1373...` | our fused TP2 + EP2 launcher with Data Direct, tuner3 and TC 106, all in HBM (the setup behind 84.8 / 588.0) | **new** |

- G-B is the like-for-like quality arm: same engine image as Hy4, same precision class, same placement rule. A
  difference in Q between H-B and G-B is a difference between the models, not between quantisations.
- G-N is GLM's best known speed setup. It is 4-bit, so its quality is reported against both G-B (the cost of NVFP4
  on GLM) and H-B.
- Not in this study: a GLM sidecar arm (needs a port of the hook to GLM's MoE layer; a follow-up only if G-B is
  close to H-B on quality), the incoai GLM-5.3 DFlash2 drafter, and NVIDIA's NVFP4 checkpoint (L6-2's).
- The two models are close in shape: 78 layers, hidden 6144, 256 routed experts top-8, expert intermediate 2048,
  MLA with sparse attention, 1M context. GLM has 3 dense layers (Hy4 1) and 40B active (Hy4 49B). Byte inventory
  for GLM is read from its safetensors headers at freeze (**TBD**) and the Grace share is computed from it.

### Reuse of Hy4 data

Hy4's Q1, Q2 and N results are per item, greedy, from frozen token IDs. FP8 Hy4 gave byte-identical logprobs on
three loads, so H-B's per-item answers are a fixed reference and are not re-run. H-A has two loads (A1, A2);
both are reported, and the pooled pair is used as in L6-4h. Hy4 speed is reused from stage 3 (10 paired rounds per
cell), with one check: **K-day**, a short H-B speed-only load in the same window as G-B (5 retained rounds at C1
and C16). If its mean is outside ±3% of stage 3's B mean at either cell, all cross-model speed ratios are reported
as "day drift, not comparable" for that cell. K-day runs only if Part 1's budget allows; if not, speed ratios
carry a note that the two models were timed on different days.

## Workloads (all from `tools/hy4/data/frozen/`, MANIFEST SHA-256 as frozen for L6-4h)

The same raw items, instructions, caps and scoring as L6-4h. What changes for GLM is rendering and tokenization:
`build_inputs.py` gets a `--model glm53` path that renders the same messages with GLM-5.3's own
`chat_template.jinja` and `tokenizer.json` (SHA-256 **TBD**) and writes a second frozen set
`frozen-glm53/` with its own MANIFEST. The harness still sends token IDs and gates on
`usage.prompt_tokens == len(ids)`.

- **S (speed).** `speed.json` (the 16 Study C prompts) and `speed_heldout.json` (16 held-out prompts), 512 output
  tokens with `ignore_eos`, temperature 0, seed 1234. C1 and C16. One warm round plus 5 retained rounds per cell per
  load. The same prompt text gives different token counts per model; the throughput metric is completion tokens
  per second, and prompt token counts per model are reported.
- **P (cold prefill).** 4,095- and 32,768-token prompts, as L6-4h.
- **N (needle).** As L6-4h: 3 documents x 3 codes at 32,512 and at 130,816 tokens.
- **Q1 (GSM8K-500)** and **Q2 (MMLU-Pro-500)**: the L6-4h items (Q2 seed 20261120, stratified), zero-shot,
  thinking off, at most 1,024 output tokens, 16 in flight, the L6-4h scoring rules.
- **Part 2 only, T (thinking on).** MMLU-Pro, the first **200** items of the frozen Q2 order (**TBD**: 200 vs a
  larger set, from Part 1's measured time per item), plus the first 200 GSM8K items, thinking on at each model's
  default effort, cap 16,384 tokens, 8 in flight. Only text after the closing think tag is scored. Needs both
  models loaded again (Hy4 has only 40 thinking-on items today).

### Thinking control for GLM (Study D amendment D-1 lesson)

Before freeze: render GLM-5.3's template for one user turn with `enable_thinking` true, false and absent, record
the prompt text and token counts, and freeze the thinking-off ending (an empty think block or the template's own
off form). Gates per request: prompt tokens equal the frozen count; thinking off, no reply contains `</?think>`;
`skip_special_tokens=false` so the gates can see the tags. L6-4h's lesson applies: if the server hides special
tokens, the gates and the answer split fail silently.

## Measures

- **Quality:** Q1 and Q2 accuracy with Wilson 95% intervals; paired differences (G - H) on the same items with
  Newcombe method-10 intervals and exact McNemar; discordant shares and the realised MDD; cap hits and empty
  answers per model. **Pre-registered secondary (the STQ1_0 lesson):** the paired Q2 difference on items where
  neither model hit the 1,024-token cap, with its count.
- **Speed:** per load and cell, mean of the 5 retained rounds with Student t95 (df 4); cross-model ratios of
  means, with a bootstrap over rounds (10,000, seed 42) for the interval. Prefill median of 3. Drafter acceptance
  and mean accepted length from `/metrics`.
- **Needles:** codes found per document and length.
- **NVFP4 cost on GLM:** Q1 and Q2 paired G-N - G-B. (Token-level drift is not measured: G-N runs in another
  engine, and drift against H is meaningless across models.)

## Decision rules

### MDD (80% power, two-sided alpha 0.05, paired, n = 500)

Q2 MMLU-Pro: **4.9 points** if 15% of items are discordant, 6.3 at 25%. Q1 GSM8K: 2.8 points at 5% discordant.
Both models are near 97% on GSM8K and near 84% on MMLU-Pro (Hy4), so this study can see a gap of about 5 points
on MMLU-Pro and no smaller. The vendor-claimed gaps are smaller than that. "No measurable difference" means the
interval includes zero; it is not a claim that the models are equal.

### Q, per benchmark (G-B vs H-B, the primary comparison)

- **"GLM higher"** / **"Hy4 higher"** if the paired 95% interval lies entirely above / below zero.
- **"No measurable difference"** otherwise, reported with the MDD.
- The same three readings for G-N vs H-B, G-N vs G-B and G-B vs H-A, as secondary comparisons.

### V, speed

Descriptive. Reported: each arm's C1 and C16 on both prompt sets, the ratios G-N / H-A (best vs best) and
G-B / H-B (matched FP8), and K-day's check. No significance test between models: different models, different loads.

### Champion (TBD at freeze, PI's call; recommendation below)

"Two-Station champion" is one model and setup we name for this hardware. Recommended rule, fixed before any GLM
number is seen:

1. If Q2 (G vs H, best setups: G-N vs H-A; and matched: G-B vs H-B) gives "no measurable difference" in both
   comparisons, the champion is the faster best setup at C16 on the held-out prompts.
2. If one model is higher on Q2 in the matched comparison and is not more than 1.5x slower at C16 in best setups,
   it is the champion.
3. If one model is higher on Q2 and the other is more than 1.5x faster at C16, the verdict is split: a quality
   champion and a speed champion, both named.

GSM8K is reported but does not decide (both near ceiling). Needles decide only if one model misses codes the other
finds. Changing any production route stays the PI's decision.

## Budget (GB300-hours: each GB300 per wall hour; RTX-hours separate)

Load to health for G-B is estimated at 50 min (704 GiB from local disk; Hy4 FP8 took about 45 min), G-N at 20 min.
Replaced by measured values once known.

| Block | Wall (est.) | GB300-h |
|---|---|---:|
| Part 1: preflight, claim, dailies stopped, fallbacks checked | 0.4 h | 0.8 |
| Part 1: G-B load, S (both sets), P, N, Q1, Q2 | 2.3 h | 4.6 |
| Part 1: G-N load, S (both sets), P, N, Q1, Q2 | 1.7 h | 3.4 |
| Part 1: K-day H-B speed-only load (optional, last) | 1.0 h | 2.0 |
| Part 1: restore reserve | 0.5 h | 1.0 |
| **Part 1 cap** | **5.9 h** | **11.8** |
| Part 2: T on H-B and G-B (two loads), T on G-N if budget | 6.0 h cap | 12 |

Order inside Part 1: G-B first (the primary question), then G-N, then K-day. Budget rule (as L6-4h): a load does
not start if elapsed + its estimate + 0.5 h restore exceeds the window or the cap; the rest is recorded as unrun.
If tonight's window is shorter than about 3 h, only G-B runs and G-N moves to the next window.

## Predictions (recorded before any run; formal values TBD at freeze)

To be produced by the L6 speed model (`tools/l6-model/hy4model.py` with a GLM-5.3 FP8 byte profile from its headers)
and hashed before the claim. The drafter's priors, for the record:
- G-B: C1 about 30 tok/s (range 15-60), C16 about 110 (40-220). Like H-B, decode is bound by Grace expert reads;
  GLM has fewer active parameters and slightly fewer Grace-resident bytes, so a little faster than H-B.
- G-N on our prompts: C1 75-90, C16 500-650 (from the catid-shape run; our prompts are shorter).
- Quality: no prior strong enough to predict a direction at a 5-point MDD; expected reading "no measurable
  difference" on Q2 for G-B vs H-B.

## Stops

- Any shard fails its SHA-256 against HF LFS: no load.
- The vLLM `hy4` image fails to load GLM-5.3 FP8 twice: G-B is unrun in that image. A different pinned vLLM image
  may be used only if named at freeze as the fallback (**TBD**); a fallback run is reported as "different engine".
- Any `check` gate fails (prompt tokens, specials, thinking): that load stops; nothing is measured.
- Thinking gate failures above 2% of a workload's requests: that workload is "not comparable" for the load.
- NCCL INIT log shows socket transport instead of IB: that load is void; twice in a row stops.
- Production needs either Station, the claim is revoked, a fallback route fails, or anyone changes a router or
  daily: stop and restore. Any step needing sudo that gets "password required": stop. A second consecutive failure
  of the same step: stop.

## Operations and interruption contract

As L6-4h: a PI-approved window with both dailies stopped and Spark fallbacks verified; a preemptible fleet claim on
both GB300s by UUID (no new arm uses the RTX PRO 6000s, so they stay unclaimed);
servers on loopback, NCCL on the direct-DAC addresses; restore on any stop, with restore time recorded.

**Before the window (no GPU use):** copy `zai-org__GLM-5.3` from the NAS archive to `/models/hf/` on both Stations
(704 GiB each; free on `/models`: left 2.6 TB, right 5.7 TB, read 2026-11-22) and check every shard against HF LFS.
Loading from the NAS would add an unknown load time and is not allowed for a timed load.

## Public outcome

A new section and data folder in RonanLabsAI/hy4-two-stations (`results/glm53/`), or a separate repo if the PI
prefers: this protocol's SHA-256 and commit URL, per-item Q1/Q2 correctness for G-B and G-N with item IDs, S round
values, P, N, gates, dispositions, GB300-hours, restore times, predictions vs measured, and the champion reading.
Launch scripts are ours, written from scratch (the fused SGLang launcher used for G-N was adapted from an unlicensed
script and must be rewritten before it can be published; until then G-N's launch is described, not published).
Private hosts, IPs, interface names, paths and UUIDs stay out.

## Pre-freeze changes and settled values (2026-11-22, before any GPU use)

1. **Checkpoints.** GLM-5.3 FP8 pinned at `ba32f8c19c8738e2c57c71798c601d477a7bf844` (HF head, 2026-10-14). Our
   NAS copy was archived 2026-09-16; it is copied to `/models/hf/zai-org__GLM-5.3` on both Stations and every one of
   the 141 shards is checked against the LFS SHA-256 of `ba32f8c1` (`data/glm53-ba32f8c1/lfs-sha256.json`) by
   `stations/l6-4g/hashcheck.sh` before any load. Its `chat_template.jinja` differs from `ba32f8c1`'s only in
   tool-result reordering, which our prompts never reach; inputs are rendered from the `ba32f8c1` template and sent
   as token IDs, so the server's template is never used. RadixArk NVFP4 pinned at `6e389189` (its weights have not
   changed since the 2026-08-28 upload; later commits are README and template), checked the same way.
2. **Thinking off for GLM.** The GLM-5.3 template has no off switch: the generation prompt always ends with an open
   `<think>`, and past assistant turns without reasoning render as `<think></think>`. Frozen thinking-off prompt:
   the default render (system line "Reasoning Effort: Max") plus `</think>`, i.e.
   `[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{USER}<|assistant|><think></think>`. Thinking on (Part 2):
   `reasoning_effort: high`, open `<think>` (Hy4's thinking-on arm also used `high`). The rendering is checked
   identical to transformers 5.15.0 `apply_chat_template` for both forms. Gate tags: `<think>`, `</think>`;
   `<|user|>` (an EOS) and the other GLM specials are stripped before scoring.
3. **Inputs.** `build_inputs.py --model glm53` writes `data/frozen-glm53/` (MANIFEST SHA-256 `33ceea25...`). Item
   IDs, order and gold answers equal Hy4's set for GSM8K, MMLU-Pro, needles and both speed sets (a test checks it).
   Rebuilding the Hy4 set gives byte-identical files (MANIFEST `59791629...`, unchanged).
4. **Harness.** `hy4bench.py` SHA-256 `649a65e1...` adds `--frozen` (input set), a GLM tag profile chosen by the
   input set, and `--engine sglang` (SGLang's detokenize needs `skip_special_tokens: false` and answers
   `text`; its completions take no `spaces_between_special_tokens`). Hy4 behaviour is unchanged. `analyze.py`
   unchanged (`810607e3...`). Tests: 47 pass (`uvx pytest tools/hy4/tests`), 7 of them new for GLM and SGLang.
5. **G-B placement.** GLM routed experts are 9.666 GB per MoE layer (4.50 GiB per Station), the same as Hy4's;
   GLM's other weights are about half of Hy4's (21 GB against 42 GB, from the shard headers). Hy4 B used 209.0 GiB
   per GB300 after offloading 166.5 GiB (37 layers), leaving 24.3 GiB of KV. Holding GB300 use equal gives about
   147.6 GiB of GLM experts in Grace, rounded up to whole layers: **OFFGB 148.5 (33 layers)**. Fallback ladder (at
   most 2 steps, each recorded): if the load fails for memory or the KV cache cannot hold one 163,840-token
   request, 157.5, then 166.5. The load stops unless vLLM reports exactly the requested GiB offloaded (parameter
   names `experts.w13_weight`/`w2_weight` and the `routed_experts.` form; only one exists). Because GLM holds 18 GiB
   less in Grace than Hy4, the matched setup is matched on GB300 memory, not on Grace bytes; reported as such.
6. **G-B flags.** `launch-glm53.sh` is `launch-hy4.sh` with GLM's model path, config hash, `--reasoning-parser
   glm47 --tool-call-parser glm47`, and `--speculative-config {"method":"mtp","num_speculative_tokens":4}`
   (NVIDIA's two-Station GLM-5.3 setting; Z.ai's card gives none). The image registers `GlmMoeDsaForCausalLM` and
   maps `glm_moe_dsa` to its MTP path; whether that path loads with `FLASHMLA_SPARSE` and the triton MoE backend is
   known only at load. If it fails twice, G-B runs with `SPEC=0` and is reported as "no drafter" (Hy4 B with its
   drafter off ran 0.919x at C1, so this costs GLM speed, not quality).
7. **G-N.** `launch-glm53-nvfp4.sh` is our October launcher (`dd1007/launch-glm53-fused-dd.sh`) with the API on
   127.0.0.1:8020, the image pinned in the file, and tuner3 + TC 106 always on. The 84.8 / 588.0 tok/s quoted in
   "Why" were decode rates from mean time per output token on catid's shape; this study's metric is completion
   tokens over round wall time, prefill included, so G-N will read lower than those figures on the same server.
8. **Order.** G-N first (its checkpoint was already local), then G-B (after the copy and hash check), then K-day.
9. **Window.** `stations/l6-4g/window.sh check|stop|restore` (run from the desk host): checks that the Station routes'
   fallbacks `analyst-local` (Spark 2) and `burst-nim` answer through LiteLLM, snapshots and stops `glmf-daily`
   (left, :8001) and `dsv41-daily` + `dsv41-daily-sidecar` (right, :8003), and restores them with image and
   health checks. LiteLLM is never edited or restarted. `check` passed 2026-11-22 11:13 PT.

## Freeze checklist

1. PI window for Part 1; dailies plan and fallbacks.
2. GLM-5.3 FP8 copied to both Stations; revision pinned; every shard hashed against HF LFS; manifest hash recorded.
   RadixArk NVFP4 revision pinned and re-hashed.
3. `vllm/vllm-openai:hy4` (digest as L6-4h) loads GLM-5.3 FP8 in a dry check: architecture `GlmMoeDsaForCausalLM`
   is registered in the image (confirmed 2026-11-22, vLLM 0.23.1rc1.dev8179); MTP support for GLM with sparse
   attention, `--max-logprobs`, and metric names confirmed.
4. G-B launch written out verbatim: expert split between HBM and Grace computed from GLM's byte inventory under the
   L6-4h rule (KV for 16 x 163,840); MTP draft length frozen.
5. G-N launch frozen (image digest, flags, overlay, tuner3, TC 106) and hashed.
6. `build_inputs.py --model glm53` written; GLM template rendered three ways; `frozen-glm53/` MANIFEST hashed;
   re-run byte-identical.
7. `hy4bench.py` and `analyze.py` changes (GLM think-tag pattern, cross-model pairing, cap-hit secondary) tested
   against a mock server on pass and gate-fail paths; hashes recorded.
8. Champion rule confirmed or changed by the PI.
9. Predictions written and hashed.
10. Part 2's T subset size fixed, or Part 2 deferred.

## Run deviations (Part 1, 2026-11-22)

- **Window and claim.** Opened 11:45 PT on the PI's instruction. Fallbacks `analyst-local` and `burst-nim`
  answered; `glmf-daily`, `dsv41-daily-sidecar`, `dsv41-daily` snapshotted and stopped 11:45-11:48; claim
  `l6-4g-hy4-vs-glm53-20261122` (both GB300s, preemptible). Hash checks: both GLM checkpoints PASS on both Stations.
- **G-N ran as frozen** (11:49-12:30 PT, ready after 458 s; every step rc 0).
- **G-B load attempt 1 failed** (12:30-12:32 PT): `FLASHMLA_SPARSE` rejected `kv_cache_dtype fp8_e4m3`, which
  vLLM chose for GLM's `auto`. Hy4 B's `auto` resolved to bf16 (269,376 tokens in 24.33 GiB). Attempt 2 (12:33 PT)
  pins `--kv-cache-dtype bfloat16` to match Hy4 B. Nothing else changed. A second failure makes G-B unrun in this
  image (Stops).

## Amendment 1 (2026-11-22 12:36 PT; PI approved in session "proposed fix accepted")

Written after G-N's Part 1 results were seen and before any G-B output or any thinking-on data for either model.

**Why.** GLM-5.3 does not reliably stay in thinking-off mode. On G-N, 31 of 500 MMLU-Pro replies (6.2%) reasoned
after the empty think block, closed it with `</think>` and answered again (GSM8K: 0 of 500). That exceeds the 2%
gate, so G-N's Q2 is "not comparable" under the frozen rule, and G-B is the same model. The thinking-off Q1/Q2
results stay in the record with their gate verdicts; Q2 thinking-off is no longer the primary question.

**New primary quality question (T, Part 2).** MMLU-Pro, all 500 frozen items (same IDs and order as Q2), thinking
on at each model's own default effort: Hy4 `high` (its template default) and GLM-5.3 `max` (its template default;
Z.ai: "for benchmark and leaderboard reproduction, keep the default max"). Greedy, cap 16,384 output tokens, 8 in
flight. Score: the text after the last closing think tag, by the Q2 rule. A reply that hits the cap without closing
its think block is wrong and counted. Gates: prompt tokens equal the frozen count; the reply closes its think block
or hits the cap. Gate failures above 2% make T "not comparable" for that load. GSM8K is not re-run thinking on
(both models are near ceiling thinking off).

**Arms.** The best setups: H-A (Hy4 FP8 + MXFP4 warm experts on the RTX PRO 6000s, the L6-4h stage-3 recipe and
rowmap) against G-N (GLM-5.3 NVFP4, SGLang). Optional, only if budget remains: H-B against G-B thinking on (each
FP8-B load needs about 3 h for 500 thinking items at its decode speed).

**Statistic and readings.** Paired G-N - H-A on the same 500 items: Newcombe method-10 interval, exact McNemar,
discordant counts, realised MDD (expected about 5 points). "GLM higher" / "Hy4 higher" / "no measurable difference"
as in Part 1. Pre-registered secondary: the same comparison on items where neither model hit the cap, with the count.
Cap hits and mean reply length per model are reported.

**Champion rule (replaces the Part 1 wording; thresholds unchanged).** Quality = T (best setups). Speed = C16 on
the held-out prompts, best setups (Part 1). (1) T "no measurable difference": the faster model is champion.
(2) One model higher on T and not more than 1.5x slower: it is champion. (3) One model higher on T and the other
more than 1.5x faster: split verdict, quality champion and speed champion named.

**Inputs and harness (before the Part 2 window).** New frozen file `mmlu_pro_think.json` in `data/frozen/` (Hy4,
`high`) and `data/frozen-glm53/` (GLM, default `max`), built by `build_inputs.py`; existing frozen files are not
touched (the Hy4 MANIFEST gains one entry). Harness: `quality --think-file mmlu_pro_think.json` runs only that set.
Both are tested and hashed before any Part 2 load.

**Budget (Part 2 window).** H-A: two sidecars plus vLLM load about 1 h, T about 1.5-2 h (500 replies at a few
thousand tokens each, about 250 tok/s at 8 in flight). G-N: load about 8 min, T about 1 h. Restore 0.25 h. About
4-4.5 h, 9 GB300-h, plus RTX-hours for H-A. Order: G-N first, then H-A.

## Results, Parts 1 and 2 (recorded 2026-11-23 PT, before any Part 3 input was built)

Window 2026-11-22 11:45-19:24 PT, claim `l6-4g-hy4-vs-glm53-20261122`, both GB300s. Raw outputs are per item and
per round; the paired statistics below were recomputed from the per-item files on 2026-11-23.

**T, thinking-on MMLU-Pro (primary under Amendment 1), 500 items, cap 16,384.**

| | H-A (Hy4, `high`) | G-N (GLM-5.3 NVFP4, `max`) |
|---|---|---|
| Correct | 436 (87.2%) | 424 (84.8%) |
| Cap hits (scored wrong) | 64 | 100 |
| Gate failures | 0 | 0 |
| Reply tokens, mean / median | 4,724 / 1,630 | 4,179 / 747 |

- G-N - H-A: **-2.4 points, 95% interval -5.2 to +0.3** (Newcombe method 10), exact McNemar p = 0.10, discordant 17
  (GLM only) against 29 (Hy4 only). Realised MDD about 3.8 points. Reading: **no measurable difference**.
- Secondary, items where neither model hit the cap (n = 377): 362 against 362, 96.0% each, difference 0.0
  (-2.0 to +2.0).
- Cap hits on one model only: GLM 59, Hy4 23. The overall gap comes from items GLM did not finish in 16,384 tokens.

**Part 1, thinking off.**

| | G-N | G-B | H-B (reused) | H-A (reused) |
|---|---|---|---|---|
| Decode C1 / C16, frozen prompts (tok/s) | 83.3 / 603.1 | 34.45 / 100.94 | 27.8 / 94.3 | |
| Decode C1 / C16, held-out prompts (tok/s) | 83.3 / 593.3 | 39.74 / 102.95 | 28.5 / 81 | 73 / 309.4 |
| Prefill 4K / 32K (tok/s) | 11,006 / 11,567 | 2,796 / 3,014 | | |
| GSM8K | 97.0% | 96.8% | 97.4% | |
| MMLU-Pro | 84.6%, not comparable | 84.8%, not comparable | 83.8% | |
| Needles | 18 / 18 | 18 / 18 | 18 / 18 | |

- GSM8K paired: G-B - H-B -0.6 (-2.4 to +1.2); G-N - H-B -0.4 (-2.2 to +1.3); G-N - G-B +0.2 (-1.3 to +1.7).
  No measurable difference in any pairing.
- Thinking-off MMLU-Pro is "not comparable" for both GLM arms: GLM reasoned after the empty think block in 31 (G-N)
  and 32 (G-B) of 500 replies, above the 2% gate (Amendment 1).
- K-day (H-B, same window): 28.18 / 96.05 tok/s at C1 / C16, within 3% of stage 3 at both cells, so cross-day speed
  ratios stand.
- G-B setup as run: 148.5 GiB of experts per Station in Grace, MTP 4 draft tokens, KV 252,992 tokens,
  `--kv-cache-dtype bfloat16` (Run deviations).

**Champion (Amendment 1 rule).** T gives no measurable difference, so the faster best setup at C16 on held-out
prompts is champion: G-N 593.3 against H-A 309.4 tok/s (1.92x). **Champion: GLM-5.3 NVFP4 in SGLang.** Hy4's point
estimate on T is higher; the record says "no measurable difference", not "equal". Coding is not covered by T;
Amendment 2 adds it.

## Amendment 2: Part 3, coding on fresh AtCoder problems (2026-11-23 PT; PI approved in session "amendment approved as writen")

Written after Parts 1 and 2 were seen and before any Part 3 problem was rendered or sent to either model.

**Why.** Parts 1 and 2 test math and knowledge. Coding is untested. The standard coding sets are older than both
models: LiveCodeBench v6 ends April 2025, LiveCodeBench Pro September 2025, SWE-bench Verified uses 2023-2024
issues. GLM-5.3 launched 2026-08-14 and Hy4 2026-11-04, so only problems published after 2026-11-04 are new to
both. AtCoder runs rated contests every week, which gives a set neither model can have seen.

**Set.** Every task of every rated AtCoder Beginner (ABC), Regular (ARC) and Grand (AGC) contest that started
after 2026-11-04 23:59 PT and ended at least 72 hours before freeze. Excluded, with counts reported: heuristic
contests (AHC), interactive tasks, and tasks dropped at test validation (below). As of 2026-11-23 that is abc483,
agc079, abc484, abc485 and arc234, 29 tasks; abc486 (2026-11-28) and abc487 (2026-12-05) join if the freeze is
after their 72-hour mark. The list, task IDs, time and memory limits and a SHA-256 of each statement are fixed
at freeze.

**Prompt.** One user message per task: the English statement as plain text (math kept as written, samples
included), then the same fixed instruction for both models: write one complete C++ program (GNU C++20) that reads
standard input and writes standard output, final program in a single ```cpp block. Rendered with each model's own
template, thinking on at each model's default effort (Hy4 `high`, GLM `max`), frozen to token IDs and hashed, as
in Amendment 1. Statements are not republished; IDs and hashes are.

**Generation.** Arms G-N and H-A only (the best setups). Four samples per task per model, seeds 1-4, at each
vendor's shipped sampling (`generation_config.json` at the pinned revisions: Hy4 temperature 0.9, top_p 1; GLM
temperature 1.0, top_p 0.95). Greedy was used in Part 2; with 30-45 tasks one greedy sample per task is too coarse,
and both vendors report coding with sampling. Output cap **32,768** tokens (Part 2 showed the 16,384 cap decided
most of the gap, and coding replies run longer). 8 in flight.

**Tests (built and frozen before any model output exists).** Per task:
1. All official samples.
2. At least 30 generated tests: small random, maximum-size random and constructed edge cases, from a generator
   written to the stated constraints, each input checked by a constraint validator.
3. Expected outputs from a reference solution transcribed from the official editorial, cross-checked against an
   independent brute force on at least 200 small random inputs (all must agree).
4. A checker for tasks that accept more than one correct output.
5. The reference must pass every test within half the time limit on the grading machine.

A task that fails steps 3-5 is dropped before any model output is generated and listed with the reason. Tests are
built by Claude (not a model under test) and hashed in the freeze manifest.

**Grading.** Take the last ```cpp block in the text after the final closing think tag. Compile with g++ 13.3
`-std=gnu++20 -O2`; run each test in a `bwrap` sandbox with no network, the task's memory limit and its stated
time limit. A sample passes if every test is correct. No code block, compile error, a cap hit or any failed test
is a fail and is counted by type.

**Gates.** Prompt tokens equal the frozen count; the reply closes its think block or hits the cap. Gate failures
above 5% of a model's replies make Part 3 "not comparable" for that load.

**Statistic and readings.** Per task, pass rate = passes / 4 for each model. Primary: mean over tasks of
(GLM - Hy4) pass rate, 95% interval by paired bootstrap over tasks (10,000 resamples, fixed seed), with an exact
sign-flip permutation p-value. "GLM higher" / "Hy4 higher" if the interval excludes zero, otherwise "no measurable
difference". With about 30-45 tasks the realised MDD is expected near 15 points; the report states it. Secondary,
descriptive: the same by tier (ABC A-D, ABC E-G, ARC and AGC), pass@4, cap hits, compile errors, mean reply length.

**Champion.** Amendment 1's champion stands for general use. Part 3 adds a coding reading. If Part 3 gives
"Hy4 higher", the verdict becomes split: GLM-5.3 NVFP4 champion overall, Hy4 champion for coding. If "GLM higher"
or "no measurable difference", the champion is unchanged. PI may change this before freeze.

**Predictions.** Written and hashed before the Part 3 window, in `tools/hy4/data/l6-4g-predictions.json`
(new `part3` key).

**Budget (Part 3 window).** H-A load about 1 h; G-N load about 8 min. Generation for 29-43 tasks x 4 samples at
about 12,000 tokens per reply: G-N about 1-1.5 h, H-A about 1.5-2.5 h. Grading runs on the Threadripper CPU and
needs no GPU. Restore 0.25 h. About 4-5.5 h on both GB300s (8-11 GB300-h) plus RTX-hours for H-A. Order: G-N
first, then H-A.

### Part 3 preparation notes (2026-11-23 PT, before any Part 3 prompt was sent to either model)

- **Tooling.** `tools/hy4/part3/`: `fetch_tasks.py` (statements, samples, limits), `sandbox.py` + `runner.c` (bwrap,
  no network, CPU time and peak RSS measured inside the sandbox), `build_tests.py` (the five test steps),
  `freeze_set.py` (writes `data/part3/SET.json`), `grade.py`, `analyze.py`. Harness: `hy4bench.py code` (4 seeds,
  vendor sampling, cap 32,768, 8 in flight); `build_inputs.py` renders `code_atcoder.json` from SET.json for both
  models. Station phases `gnc` (G-N) and `hac` (H-A) in `stations/l6-4g/run.sh`. Tests in
  `tools/hy4/tests/test_part3.py` (56 of 56 pass across the suite).
- **Tests.** Built by Claude subagents from the official editorials, one per contest, each verified by
  `build_tests.py`. 28 of 29 tasks ok (30-60 generated tests each, 200-seed brute cross-check, deliberate wrong
  solutions fail). Every reference grades AC through `grade.py`. `agc079_d` is dropped for now: its reference took
  4.1 s CPU on one test against the 4.0 s budget (half its 8 s limit) while the Threadripper ran at load 300-400
  from other sessions; it is re-checked on a quiet host at the final freeze. Notes per task in `NOTES.md`; to
  review: `abc483_e` (editorial formula, keys sorted instead of hashed; byte-identical outputs),
  `abc483_f`, `abc483_g`, `abc485_b/f/g`, all four `arc234` tasks (references written from editorial prose).
- **Grading guards (fixed now, before any output).** The grader refuses to start above 1-minute load 32, and a
  TLE counts only if it repeats on an immediate re-run of that test. Statements are stored with LF line ends
  (some AtCoder samples are CRLF; sample files keep their bytes).
- **Generated tests are not in git** (about 1 GB); every file's SHA-256 is in its task's `BUILD.json` and the
  grader re-checks them. Backup: `workspace/l6-4g-part3/tasks-provisional-20261123.tar.zst`.
- **Provisional freeze.** SET.json and `code_atcoder.json` are built for the 28 tasks so the chain could be tested.
  The final freeze, just before the window, adds every eligible contest that has ended 72 h earlier (abc486
  after 2026-12-01, abc487 after 2026-12-08), re-checks agc079_d, re-runs `freeze_set.py` and `build_inputs.py`,
  and records the hashes here.
- **Predictions** added under `part3` in `tools/hy4/data/l6-4g-predictions.json` (file SHA-256 before the new key:
  5b101f28...; after: a4573b06...).

### Part 3 final freeze (2026-11-22 22:46 PT, before any Part 3 prompt was sent to either model)

- **Set: 29 tasks, 0 excluded.** abc483 a-g, agc079 a-d, abc484 a-g, abc485 a-g, arc234 a-d. No interactive tasks;
  no new eligible contest (abc486 is 2026-11-28).
- **Deviation, PI decision in session ("Keep all 5 contests").** Amendment 2 is internally inconsistent: its rule
  says "ended at least 72 hours before freeze", and its own list names abc485 and arc234. Checked on atcoder.jp,
  abc485 ended 2026-11-21 05:40 PT (72 h mark 2026-11-24 05:40) and arc234 ended 2026-11-22 06:30 PT (72 h mark
  2026-11-25 06:30), so neither meets the clause at this freeze. The explicit list wins over the 72 h clause. Their
  tests were already built from the official editorials, and no model had seen any Part 3 prompt.
- **agc079_d re-check** at 1-min load 27 (22:44 PT): ok, reference max CPU 3.43 s against the 4.0 s budget (TL 8 s),
  37 generated tests, 200-seed brute cross-check agrees. It joins the set.
- **NOTES.md doubts reviewed** (abc483_e/f/g, abc485_b/f/g, arc234_a-d): accepted. The prose-derived references
  agree with independent brute forces on 200 small seeds each, and abc483_e's literal editorial transcription
  matches all 41 expected outputs.
- **Checks.** `pytest tools/hy4/tests`: 56 of 56 pass. Reference self-grade through `grade.py` at load 10: 29 of 29
  AC (max CPU 3.19 s, agc079_d). All other frozen files byte-identical to the provisional freeze; only
  `code_atcoder.json`, both MANIFESTs, SET.json and agc079_d/BUILD.json changed.
- **Hashes (SHA-256).**
  - `data/part3/SET.json`: 55c81a7407e661cb1a2e8b9a0073e5183c094db3c1eaec76b6e03c778c58e436
  - `data/frozen/code_atcoder.json` (Hy4): 6af473309161ecc415ce0bb2f47fde988f820a95ea07e46f5d82c04ecef45942
  - `data/frozen/MANIFEST.json`: bfb8cc97a60a48241a0c8d0f08b49268f42c7676fdf31d13cfce03f2d1138c0e
  - `data/frozen-glm53/code_atcoder.json` (GLM): 5caf2183e27613c8cf6730f5f00118fb7710ca0af4f4b2731c38b60dacea463e
  - `data/frozen-glm53/MANIFEST.json`: e32af672debc9a05cc11ed97b128e33bd32df18175d6edf9d758abbe2cd4cab0
  - `data/l6-4g-predictions.json`: 2bcd742b10342b159deabe84c1b3bdef5d793071230d1543337352cb7265eef4 (only
    `part3.set_at_writing` changed; predicted values unchanged)

## Amendment 3: Part 3b, the same tasks at a 65,536-token cap (2026-11-23 17:35 PT; PI pre-authorized further rounds in session: "if indicated, can another rounds")

Written after the Part 3 outputs of both arms were graded and before any Part 3b output exists. Exploratory and
secondary: the Part 3 reading at 32,768 stands as recorded and is not replaced; the champion rule is not
re-applied to Part 3b.

**Why.** At 32,768, every failed reply of both models was a cap hit (G-N 65 of 116, H-A 55 of 116); no reply
that finished was wrong, and no reply failed to compile. The Part 3 difference is therefore a difference in how
often each model finishes thinking inside 32,768 tokens. Part 3b asks whether the extra replies pass when given
twice the room.

**Why 65,536 and not 131,072.** H-A's KV cache holds 269,376 tokens and G-N's 471,232, so at 131,072 neither could
keep 8 long replies in flight and the round would take an estimated 8-10 h; at 65,536 H-A fits about 4 and G-N
about 7.

**Set, prompts, tests, grading.** Unchanged: the frozen 29 tasks, the same `code_atcoder.json` token IDs and
hashes, the same tests and `grade.py`. No re-freeze.

**Generation.** Phases `gnc64` then `hac64` in `stations/l6-4g/run.sh` (identical to `gnc`/`hac` except
`--code-cap 65536`), same loads and recipes. Seeds 1-4, vendor sampling, 8 in flight. All 116 samples per model
are redrawn; re-drawing only the capped samples would bias toward replies that happen to finish early.

**Time box.** At 2026-11-24 08:00 PT any running generation is stopped. Jobs are seed-major, so a stopped arm
holds whole seeds plus part of the next; the paired analysis then uses only the seeds complete in both arms.
**Time box removed** 2026-11-23 19:40 PT, before G-N finished, on PI instruction ("Just finish it, dont cut off
unless a problem"): both arms run to completion; the stop rules for failures still apply.

**Statistic and reading.** As Amendment 2 (paired bootstrap over tasks, 10,000 resamples, fixed seed; exact
sign-flip p), labelled "Part 3b, 64K cap, exploratory". Also: cap hits at 64K, reply length, and per task the
32K and 64K pass rates side by side.

## Results, Part 3 (32,768 cap; recorded 2026-11-23 PT)

Window 2026-11-23 11:56-17:33 PT, claim `l6-4g-part3-hy4-vs-glm53-20261123`, both GB300s. Dailies had been
restored after Part 2 and were stopped again with `window.sh stop` at 11:56-12:00 PT (fallbacks OK). G-N
12:00-14:09 PT (load 16 min, generation 113 min); H-A 14:10-17:33 PT (load 11 min, generation 192 min). Graded on
the Threadripper at 1-min load 0.3 and 0.2. Raw outputs: `workspace/l6-4g-runs/{gnc-20261123-120035,
hac-20261123-141047}`; analysis `workspace/l6-4g-runs/part3-analysis.json`.

| | G-N (GLM-5.3 NVFP4, `max`) | H-A (Hy4, `high`) |
|---|---|---|
| Replies passing all tests (of 116) | 51 (44.0%) | 61 (52.6%) |
| Cap hits (all other replies) | 65 | 55 |
| Wrong answer, TLE, RE, compile error, no code | 0 | 0 |
| Gate failures | 0 | 0 |
| Tasks with at least one pass (pass@4, of 29) | 14 | 17 |
| Mean reply tokens; median of passing replies | 22,420; 4,711 | 20,551; 7,202 |
| ABC A-D (12 tasks), mean pass rate | 85.4% | 95.8% |
| ABC E-G (9 tasks) | 27.8% | 41.7% |
| ARC and AGC (8 tasks) | 0% | 0% |

- **Primary:** GLM - Hy4 mean per-task pass rate **-8.6 points, 95% bootstrap interval -18.1 to -1.7**; exact
  sign-flip p = 0.0625. Five tasks differ, all toward Hy4 (abc483_e 0.50 vs 0.75, abc483_g 0 vs 0.50, abc484_c 0.25
  vs 0.50, abc484_d 0 vs 1.00, abc485_e 0 vs 0.50); none toward GLM. Realised MDD 11.7 points.
- **Reading (Amendment 2 rule, interval):** **"Hy4 higher".** Champion rule: split verdict, GLM-5.3 NVFP4 champion
  overall (Amendment 1), Hy4 champion for coding. Stated with it: the interval and the permutation test disagree
  (p 0.06 with only 5 discordant tasks, the smallest p possible with 5 being 0.0625), so the result is at the edge.
- **Every failure is a cap hit.** No reply that closed its think block failed a test. The Part 3 difference is how
  often each model finishes inside 32,768 tokens, not how often its finished code is wrong. Capped replies were
  still reasoning (not looping) when cut off; 15 G-N and 9 H-A capped replies had already written a cpp block
  inside the think block, which the rule does not grade.
- **Predictions vs measured:** overall Hy4 60% / GLM 63% predicted, 52.6% / 44.0% measured (both inside the
  45-80% band for Hy4; GLM just below it). By tier: ABC A-D 95/95 predicted, 96/85; ABC E-G 55/60, 42/28; ARC-AGC 15/20, 0/0.
  Reading predicted "no measurable difference" (about 70%), slight lean GLM: wrong in direction. Cap hits
  predicted under 15% for both: measured 56% (GLM) and 47% (Hy4); the "GLM >= Hy4" part held. Gate failures under
  2%: held (0).
- Amendment 3 (above) adds an exploratory 65,536-cap round, Part 3b, for the cap question.

## Correction to Results, Part 2 (found 2026-11-23 ~18:30 PT while drafting the write-up; needs PI review)

Amendment 1 says "A reply that hits the cap without closing its think block is wrong and counted." The run-time
scorer did not do that: `hy4bench.py answer_text` returns the whole text when no closing think tag exists, so
`score_mmlu` searched the unfinished reasoning for "answer is (X)" or a last "(X)" and counted a match. 44 G-N and
28 H-A capped replies were scored correct that way; the "Cap hits (scored wrong)" row in Results, Parts 1 and 2
was wrong. One H-A reply closed its think block and then hit the cap in the answer; it is scored on its answer,
as the rule allows. Raw files are unchanged; `tools/hy4/part2_rescore.py` applies the rule as written
(output `workspace/l6-4g-runs/part2-rescore.json`).

| T, thinking-on MMLU-Pro, 500 items | H-A (Hy4) | G-N (GLM-5.3) |
|---|---|---|
| Correct, as recorded at run time | 436 (87.2%) | 424 (84.8%) |
| **Correct, Amendment 1 rule** | **408 (81.6%)** | **380 (76.0%)** |

- G-N - H-A: **-5.6 points, 95% interval -9.0 to -2.2** (Newcombe method 10), exact McNemar p = 0.0015,
  discordant 23 (GLM only) against 51 (Hy4 only). Reading: **"Hy4 higher"** (was "no measurable difference").
- Unchanged: items where neither model hit the cap, n = 377, 362 against 362. The gap is still the items GLM did not
  finish inside 16,384 tokens; the correction stops counting a guess from unfinished reasoning as an answer.
- **Champion, Amendment 1 rule applied to the corrected T.** Hy4 higher on T, and G-N is 1.92x faster at C16 on
  held-out prompts (above 1.5x): rule (3), **split verdict: Hy4 quality champion, GLM-5.3 NVFP4 speed champion.**
  This replaces "Champion: GLM-5.3 NVFP4 in SGLang" in Results, Parts 1 and 2.
- Effect on Part 3: Amendment 2's champion paragraph was written on the old reading. Its "split" case (Hy4 higher
  on coding) now agrees with the corrected T: Hy4 higher on both quality measures, GLM about 1.9x faster.
- Part 3 grading is not affected: `grade.py` fails any reply that hits the cap without closing its think block.
- Part 1 (thinking off) is not affected: those replies have no think block and the 1,024 cap applies to the answer.

## Results, Part 3b (65,536 cap; exploratory; recorded 2026-11-24 PT)

Same window and claim as Part 3, extended under Amendment 3. G-N 17:35-21:08 PT (load 16 min, generation 197
min); H-A 21:09-02:25 PT (load 11 min, generation 305 min). Both arms ran to completion, 116 of 116 each. Raw
outputs pulled with sha256 matched: `workspace/l6-4g-runs/{gnc64-20261123-173542,hac64-20261123-210933}`;
analysis `workspace/l6-4g-runs/part3b-analysis.json`. Claim released 2026-11-24 02:30 PT; dailies left stopped
on PI instruction (restore with `window.sh restore`).

| | G-N 64K | H-A 64K | (G-N 32K | H-A 32K) |
|---|---|---|---|---|
| Replies passing all tests (of 116) | 68 (58.6%) | 83 (71.6%) | 51 | 61 |
| Cap hits | 46 | 32 | 65 | 55 |
| Compile errors | 2 | 1 | 0 | 0 |
| Wrong answer, TLE, RE, no code | 0 | 0 | 0 | 0 |
| Tasks with at least one pass (pass@4, of 29) | 18 | 24 | 14 | 17 |
| Mean reply tokens; median of passing replies | 38,843; 13,363 | 31,411; 12,208 | 22,420; 4,711 | 20,551; 7,202 |
| ABC A-D (12 tasks), mean pass rate | 97.9% | 100% | 85.4% | 95.8% |
| ABC E-G (9 tasks) | 58.3% | 69.4% | 27.8% | 41.7% |
| ARC and AGC (8 tasks) | 0% | 31.3% | 0% | 0% |

- **Part 3b, 64K cap, exploratory:** GLM - Hy4 mean per-task pass rate **-12.9 points, 95% bootstrap interval
  -24.1 to -4.3**; exact sign-flip p = 0.016. Ten tasks differ, 9 toward Hy4 and 1 toward GLM. Realised MDD 14.6
  points. Reading: "Hy4 higher". The champion rule is not re-applied (Amendment 3).
- **Compile errors** (the only non-cap failures in either round): G-N abc483_f seed 4 (`#pragma GCC target`
  always_inline mismatch under g++ 13) and abc484_g seed 4 (undeclared `smmask`, a typo); H-A abc484_f seed 3
  (a user function `identity()` ambiguous with `std::identity` under `using namespace std`). Still no finished,
  compiling program gave a wrong answer, a time-out or a runtime error.
- **ARC.** Hy4 solved arc234_a and arc234_b on all 4 seeds and arc234_c and arc234_d on 1 seed each; GLM solved no
  ARC or AGC task at either cap. Neither model solved an AGC task.
- **Where the extra room went.** GLM's gains were on tasks Hy4 already led at 32K (abc483_e, abc483_g, abc484_c,
  abc484_d, abc485_e), bringing most to or near Hy4's level; Hy4's gains were new tasks (abc483_f, abc484_g,
  abc485_f, four ARC). Median passing reply roughly doubled for Hy4 and nearly tripled for GLM: many 64K passes
  needed more than 32K tokens of thinking.

Per task, tasks whose pass rate is not the same in all four cells (GLM 32K, GLM 64K, Hy4 32K, Hy4 64K):

| Task | G-N 32K | G-N 64K | H-A 32K | H-A 64K |
|---|---|---|---|---|
| abc483_e | 0.50 | 1.00 | 0.75 | 1.00 |
| abc483_f | 0.00 | 0.00 | 0.00 | 0.25 |
| abc483_g | 0.00 | 0.75 | 0.50 | 1.00 |
| abc484_c | 0.25 | 1.00 | 0.50 | 1.00 |
| abc484_d | 0.00 | 0.75 | 1.00 | 1.00 |
| abc484_f | 1.00 | 1.00 | 1.00 | 0.75 |
| abc484_g | 0.00 | 0.50 | 0.00 | 0.75 |
| abc485_e | 0.00 | 1.00 | 0.50 | 1.00 |
| abc485_f | 0.00 | 0.00 | 0.00 | 0.50 |
| arc234_a | 0.00 | 0.00 | 0.00 | 1.00 |
| arc234_b | 0.00 | 0.00 | 0.00 | 1.00 |
| arc234_c | 0.00 | 0.00 | 0.00 | 0.25 |
| arc234_d | 0.00 | 0.00 | 0.00 | 0.25 |

All other tasks: 1.00 in all four cells (the remaining ABC A-D tasks, abc484_e) or 0.00 in all four (abc485_g,
agc079_a-d).
