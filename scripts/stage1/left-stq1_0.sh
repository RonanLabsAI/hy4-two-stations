#!/usr/bin/env bash
# L6-4h stage 1, station-left, rung STQ1_0 (215 GiB), GB300 only. DRAFT, NOT RUN.
# Placement (hy4model S1): all 78 layers + KV (163,840 tokens, ~15.6 GB F16) on the GB300; predicted free ~10.9 GiB.
RUNG=stq1_0 STATION=left
GGUF=/models/Hy4-GGUF/Hy4-STQ1_0.gguf
GGUF_SHA=ae822c4cd227a89305b562958b7ac36ff41ce330c0396a8e18ea9de0bba3cd63
source "$(dirname "$0")/common.sh"
VISIBLE=$GB300
preflight "$GB300"; sampler_start
serve -ngl 999 -sm none -mg 0
measure
