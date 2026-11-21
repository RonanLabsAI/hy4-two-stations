#!/usr/bin/env bash
# Fetch the small tencent/Hy4-FP8 files the harness needs (tokenizer, chat template, config) at the revision used,
# and check their SHA-256. Only needed to rebuild data/frozen/ with build_inputs.py; the frozen token IDs are committed.
# The files are under the Tencent Hy License 1.0 (see the downloaded LICENSE), which is why they are not in this repo.
set -euo pipefail
REV=d063b5853ecf56e0776266dab28305fc17a137cc
D="$(dirname "$0")/data/hy4-fp8-d063b585"; mkdir -p "$D"; cd "$D"
for f in chat_template.jinja config.json generation_config.json LICENSE model.safetensors.index.json tokenizer_config.json tokenizer.json; do
  curl -sSfL -o "$f" "https://huggingface.co/tencent/Hy4-FP8/resolve/$REV/$f"
done
sha256sum -c <<'SUMS'
11b12531403da5726b7e1f6003f479043b05999e6746d39af949cbcdfd715c35  chat_template.jinja
2ae591dd0bc9a9a053d4655743ff8f587e5fa7c5af53f6b93df1661ae0c8d423  config.json
48febb029b014973f782b82e0272a139660476deee657fde5100fa1a059d4f98  generation_config.json
52755a6786f2f4b83b48cd7246aa9e34f7aac98563e4e60808ab02390799fdbd  LICENSE
19d5132bf5f198d57e357d719bad36880b212d4f909d949f4f16b3e6f170518f  model.safetensors.index.json
7efd9a50a889d164d34f5bb34bd257822feb11117e40fe4c292b24719648d7ad  tokenizer_config.json
2e3f3d22065c2be5765ce99ab2410411ae338b32ce809ea20dcdb61d00198e83  tokenizer.json
SUMS
