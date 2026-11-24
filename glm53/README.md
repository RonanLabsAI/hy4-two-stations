# GLM-5.3 vs Tencent Hy4 on two DGX Stations

RonanLabs study L6-4g, run 2026-11-22 to 2026-11-24 (Pacific time), on the same two DGX Station GB300s and with the
same harness as the Hy4 study in this repo. The plan, every amendment (dated) and every result are in
`plan/prereg-l6-4g-hy4-vs-glm53.md`. It was written before the runs; the 64K coding round (Amendment 3) was added
after the 32K results were graded and is exploratory.

## Setups

- **GLM-5.3 NVFP4** (`RadixArk/GLM-5.3-NVFP4`) in SGLang, TP2 + EP2 across both Stations, all in HBM.
- **Hy4 FP8** (`tencent/Hy4-FP8`) in vLLM with the RTX PRO 6000 MXFP4 expert sidecar (`../sidecar`), TP2 + EP2.
- Each is that model's fastest setup on this hardware; they are not the same precision. A matched-FP8 GLM arm
  (same vLLM image as Hy4) is in Part 1.

## Results

| | GLM-5.3 NVFP4 | Hy4 FP8 + sidecar |
|---|---|---|
| Decode, 1 user / 16 users, held-out prompts (tok/s) | 83.3 / 593 | 73.4 / 309 |
| MMLU-Pro, thinking on, 500 items, 16K cap | 76.0% (100 hit the cap) | 81.6% (64 hit the cap) |
| Coding, 116 tries, 32K cap: passed | 51 | 61 |
| Coding, 116 tries, 64K cap: passed (exploratory) | 68 | 83 |

- **MMLU-Pro thinking on:** GLM - Hy4 -5.6 points (95% CI -9.0 to -2.2), exact McNemar p = 0.0015. On the 377
  items where neither model hit the cap: 362 vs 362. This is the corrected score: the run-time scorer counted
  answers it found inside unfinished reasoning on capped replies, against the plan's rule. `tools/part2_rescore.py`
  applies the rule; see "Correction to Results, Part 2" in the plan. Output: `results/part2-rescore.json`.
- **Coding:** 29 AtCoder tasks (ABC 483-485, ARC 234, AGC 079) from contests after Hy4's 2026-11-04 release, 4 seeds
  each, C++ (g++ 13), each graded on at least 30 generated tests in a sandbox. Per-task paired bootstrap:
  - 32K: GLM - Hy4 -8.6 points (95% CI -18.1 to -1.7), sign-flip p = 0.0625, 5 tasks differ, all toward Hy4.
  - 64K: -12.9 points (-24.1 to -4.3), p = 0.016, 9 of 10 differing tasks toward Hy4. Hy4 solved four ARC tasks
    at least once; GLM solved no ARC or AGC task at either cap.
  - Every failure at 32K was a reply that hit the cap while still thinking. At 64K, 3 replies did not compile; no
    compiling program gave a wrong answer.
- **Matched FP8 (Part 1):** GSM8K 96.8% (GLM) vs 97.4% (Hy4), no measurable difference; GLM FP8 34.5 / 101 tok/s
  vs Hy4 FP8 27.8 / 94.3.

Reading: split. Hy4 gives better answers with thinking on; GLM-5.3 NVFP4 is 1.9x faster at 16 users.

## Files

- `plan/`: the plan with amendments and results; `SET.json` (task IDs, titles, statement and test-build hashes;
  statements are not republished); predictions written before the runs.
- `results/<run>/`: per-run summaries; `graded.jsonl` per coding run (one row per reply: task, seed, verdict,
  tokens). `part2-rescore.json`, `part3-analysis.json` (32K), `part3b-analysis.json` (64K).
- `tools/`: harness (`hy4bench.py`), coding grader and sandbox, analysis, rescore, Station run script.
- `figures/`: the three charts.
