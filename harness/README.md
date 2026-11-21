# harness: Hy4 benchmark and quality harness

Protocol: `../plan/prereg-l6-4h-hy4.md`.

| File | What |
|---|---|
| `hy4bench.py` | Harness for llama-server or vLLM (stdlib). `check`, `speed`, `prefill`, `needle`, `quality`, `logprobs`, `all`. |
| `analyze.py` | Offline: `drift` (top-k KL vs the FP8-B reference), `paired` (Newcombe + McNemar), `abba` (paired log-ratio t95). |
| `build_inputs.py` | Freezes every prompt as token IDs from the pinned tokenizer and chat template -> `data/frozen/`. |
| `fetch_raw.sh` | Downloads the public raw datasets to `data/raw/` (git-ignored; hashes checked by `build_inputs.py`). |
| `fetch_model_files.sh` | Downloads the tokenizer, chat template and config of `tencent/Hy4-FP8` at the revision used, with hashes. |
| `data/frozen/` | Every prompt as token IDs, with `MANIFEST.json` hashes. Committed, so the harness runs without the model files. |
| `tests/` | Mock llama-server/vLLM and tests: `uvx pytest harness/tests`. |

```sh
harness/fetch_raw.sh
uv run --no-project --with tokenizers==0.23.3 --with jinja2==3.1.6 --with pyarrow==26.0.1 python -I harness/build_inputs.py
uvx pytest -q harness/tests
python3 -I harness/hy4bench.py all --base http://127.0.0.1:18080 --engine llamacpp --rung stq1_0 --out runs/x --think
python3 -I harness/analyze.py drift runs/fp8-b/bench runs/x/bench
```

Every request is a token-ID prompt to `/v1/completions`; token counts come only from server `usage`; every
request is gated on `usage.prompt_tokens == len(ids)`. Logprobs use the one mode both engines support: the
generated-token top-k of a 1-token greedy completion (llama-server `/completion` `n_probs`; vLLM `logprobs` with
`--max-logprobs 64`), at identical positions for every rung.
