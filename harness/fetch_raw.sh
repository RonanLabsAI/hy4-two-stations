#!/usr/bin/env bash
# Fetch the public raw datasets into tools/hy4/data/raw (git-ignored); build_inputs.py checks their SHA-256.
set -euo pipefail
cd "$(dirname "$0")/data/raw"
curl -sSfL -o gsm8k-test.parquet     https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet
curl -sSfL -o mmlu-pro-test.parquet  https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet
curl -sSfL -o wikitext2-test.parquet https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet
curl -sSfL -o humaneval-test.parquet https://huggingface.co/datasets/openai/openai_humaneval/resolve/main/openai_humaneval/test-00000-of-00001.parquet
sha256sum *.parquet
