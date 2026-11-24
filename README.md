# Tencent Hy4 (770B MoE) on two DGX Stations

RonanLabs study L6-4h, run 2026-11-20 to 2026-11-21 (Pacific time). Every number below is computed from files in
`results/`, with `harness/analyze.py` or `figures/make_figures.py`.

**Hardware:** two NVIDIA DGX Station GB300s. Each has one GB300 GPU, one RTX PRO 6000 Blackwell (Max-Q) and Grace
CPU memory. The two Stations are joined by a direct network cable.

**Software:**
- FP8 runs on vLLM (`vllm/vllm-openai:hy4`) with expert parallel 2 and tensor parallel 2 across both Stations,
  with 166.5 GiB of experts in Grace memory and Hy4's built-in drafter at 5 tokens per step.
- The GGUF versions run on a llama.cpp fork at `f8d4cc414` with the STQ1_0 patch, one Station each.

**Follow-on study:** GLM-5.3 vs Hy4 on the same Stations (L6-4g): speed, MMLU-Pro with thinking on, and fresh AtCoder coding. See [`glm53/`](glm53/).

## Results

### RTX PRO 6000 expert sidecar (A) vs plain FP8 (B)

The sidecar keeps the 4,283 most-used Grace-resident experts per Station on the RTX PRO 6000 in MXFP4 and computes
them there. Order: A-B-B-A over four model loads, 10 paired rounds per cell.

| | B (FP8) | A (sidecar) | A / B (95% CI) |
|---|---|---|---|
| 1 user, tok/s | 27.8 | 69.9 | 2.51 (2.48-2.54) |
| 16 users, tok/s | 94.3 | 343.8 | 3.64 (3.58-3.71) |
| Prefill 4K, tok/s | 2,071 | 3,937 | 1.9 |

| Quality vs FP8 | A1 | A2 |
|---|---|---|
| GSM8K, paired change in points (95% CI) | 0.0 (-1.0, +1.0) | -0.2 (-1.4, +1.0) |
| MMLU-Pro, paired change in points | +1.2 (-0.8, +3.2) | -0.4 (-2.5, +1.7) |
| Needles found (32K and 128K) | 18/18 | 18/18 |
| Mean KL from FP8, nats | 0.083 (0.065-0.104) | 0.086 (0.069-0.105) |
| Top-1 agreement with FP8 | 93.8% | 93.7% |
| Extra prediction loss, nats | -0.005 (-0.020, +0.010) | -0.009 (-0.030, +0.011) |

FP8 gave byte-identical log-probabilities on three separate loads. Two loads of the identical sidecar setup differ
from each other by KL 0.081 and 93.3% top-1 agreement, about as much as either differs from FP8, so the drift is
mostly run-to-run variation in the sidecar path rather than lost accuracy (found after the run; see below).

### Compressed GGUF versions vs FP8

| Version | Placement | 1 user tok/s | 16 users tok/s | GSM8K | MMLU-Pro | KL from FP8 | Top-1 |
|---|---|---|---|---|---|---|---|
| FP8 | 2 Stations, vLLM | 27.6 | 93.2 | 97.4% | 83.8% | 0 | 100% |
| STQ1_0 | GB300 | 25.1 | 78.1 | 97.4% | 79.8% | 0.569 | 80.6% |
| IQ2 | GB300 + RTX PRO 6000 | 18.0 | 63.2 | 96.8% | 83.4% | 0.394 | 83.6% |
| Q4_K_M | GB300 + RTX PRO 6000 + Grace | 14.2 | 46.2 | 97.0% | 85.0% | 0.146 | 91.5% |

Every version found 18 of 18 needles at 32K and 128K.

### Hy4's drafter

Time per verify step, 5 drafted tokens: 114.9 ms at 2K context, 115.1 at 16K, 116.3 at 128K, 116.4 at 512K.
With the drafter off, plain decode runs at 25.4 tok/s at 1 user, against 27.6 with it on.

### Follow-ups (run 2026-11-21 12:50 to 19:41 PT, plan change 8)

**Held-out prompts.** 16 prompts the expert list never saw (8 encyclopedia requests on WikiText-2 titles, 8 HumanEval
completions), 4-bit sidecar vs FP8, A-B-B-A:

| | FP8 | 4-bit sidecar | A / B (95% CI) | Original prompts |
|---|---|---|---|---|
| 1 user, tok/s | 28.5 | 73.4 | 2.58 (2.56-2.59) | 2.51 |
| 16 users, tok/s | 81.0 | 309.4 | 3.82 (3.78-3.86) | 3.64 |

**8-bit (MXFP8) sidecar.** 2,206 warm experts per Station instead of 4,283. Output check before timing: cosine of at
least 0.9989 on every layer of both ranks.

| | FP8 | 8-bit sidecar | A / B (95% CI) |
|---|---|---|---|
| 1 user, tok/s | 27.8 | 47.9 | 1.72 (1.71-1.74) |
| 16 users, tok/s | 94.2 | 165.0 | 1.75 (1.73-1.78) |
| Held-out, 1 user / 16 users | 28.5 / 81.0 | 46.5 / 139.7 | |

| 8-bit sidecar vs FP8 | A8-1 | A8-2 |
|---|---|---|
| GSM8K, paired change in points | -0.4 (-1.7, +0.8) | -0.2 (-1.3, +0.8) |
| MMLU-Pro, paired change in points | +0.6 (-1.5, +2.7) | +1.0 (-1.0, +3.0) |
| Needles | 18/18 | 18/18 |
| Mean KL from FP8 | 0.080 (0.061-0.101) | 0.073 (0.057-0.092) |
| Top-1 agreement | 94.0% | 94.0% |
| Extra prediction loss, nats | +0.001 (-0.015, +0.017) | +0.009 (-0.010, +0.028) |

**Drift is mostly run-to-run variation** (exploratory, added after seeing results):

| Comparison | Mean KL | Top-1 agreement |
|---|---|---|
| FP8 vs FP8 (three loads) | 0 | 100% |
| 4-bit sidecar vs FP8 | 0.083 | 93.8% |
| 4-bit sidecar vs itself (two loads) | 0.081 | 93.3% |
| 8-bit sidecar vs FP8 | 0.080 | 94.0% |
| 8-bit sidecar vs itself (two loads) | 0.085 | 93.5% |

The 8-bit sidecar's layers match FP8 far more closely than the 4-bit ones (0.999 vs 0.98 cosine), yet both drift the
same amount, and each drifts from itself as much as from FP8. The cause of the run-to-run variation is not yet known.
On the plan's two tolerance tiers (change 8), every sidecar load misses the lossless tier on drift; the 8-bit loads
meet the near-lossless tier except A8-2's prediction-loss bound (+0.028 against +0.020).

No sidecar request timed out in any load.

## How the plan changed

The plan (`plan/prereg-l6-4h-hy4.md`) was written before the first run and never formally frozen. It was amended 8
times during the study, and each amendment is dated in the section "Pre-freeze changes". Two were made after seeing
a result:
- **Change 7:** the first sensitivity control (NCCL Simple protocol) did not slow anything down. It was replaced
  with the drafter turned off.
- **Change 8:** the follow-up design and a second, looser drift tolerance, written after stage 3.

Timestamps are local and not externally verifiable. Analyses marked "exploratory" in `results/analysis/out/` were
added after seeing results.

## Layout

| Path | What |
|---|---|
| `plan/` | The study plan with its dated amendments. |
| `harness/` | Benchmark harness, analysis, frozen token-ID inputs, tests. See `harness/README.md`. |
| `scripts/` | The exact shell scripts run on each Station (`stations/`) and the stage-1 launchers (`stage1/`). |
| `results/<station>/runs/` | Per-load results: `bench/*.jsonl` (every request) and `*.summary.json`, plus gates, launch arguments and sidecar stats. `attempt1` holds the first stage-1 attempt, stopped by its gates. |
| `results/analysis/out/` | Paired tests, drift, A-B-B-A ratios. |
| `figures/` | The charts and the script that draws them. |
| `sidecar/` | The RTX PRO 6000 warm-expert sidecar: vLLM hook, sidecar server, rowmaps, launch scripts (`launch/hy4/`), checks. Needs b12x at run time. |

Server logs are not included. The model and raw datasets are not included: `harness/fetch_raw.sh` and
`harness/fetch_model_files.sh` download them, with hash checks.

## Reproduce the analysis

```sh
uvx pytest -q harness/tests
python3 -I harness/analyze.py drift results/station-left/runs/fp8b-20261121-010950-off166.5/bench \
    results/station-left/runs/s3-A1-20261121-082643/bench
uv run --no-project --with matplotlib==3.9.2 python -I figures/make_figures.py
```

## Licences

- Code and documentation in this repo: Apache License 2.0 (`LICENSE`), including `sidecar/` (see `sidecar/NOTICE`).
- The frozen inputs are token IDs derived from:
  - GSM8K (MIT);
  - MMLU-Pro (MIT);
  - HumanEval (MIT);
  - WikiText-2 (CC BY-SA). `logprob_corpus.json` and the encyclopedia prompts in `speed_heldout.json` are derived
    from WikiText-2 and carry its licence.
- Model outputs in `results/` were generated by Tencent Hy4, which is under the Tencent Hy License 1.0.
