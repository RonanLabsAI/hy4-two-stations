#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""E3 bring-up gate: real-prompt cosine of the sidecar's warm rows vs marlin on the same experts, every warm layer.

  cos_gate.py send URL MODEL GSM8K.jsonl [N=6]       # N real few-shot prompts of ~3K tokens (eager prefill)
  cos_gate.py parse LOG_RANK0 LOG_RANK1 [ROWMAP]     # summarise "E3 check" lines; PASS = every warm layer on every
                                                     # rank has >= 1 check and min cos >= MIN_COS, and all ok=1
  MIN_COS (env, default 0.999 = DS-V4, same MXFP4 weights on both sides). Hy4 warm experts are REQUANTISED from block
  FP8 (sidecar/fp8_convert.py), so the GB300 reference and the sidecar no longer share weights: the offline gate
  (tools/offline_gate.py, 2026-11-20) puts the expected per-call cos at ~0.979-0.986 for mxfp4 and ~0.997 for mxfp8.
  Use MIN_COS=0.97 (mxfp4) / 0.995 (mxfp8) for Hy4: a layout or scale bug gives cos near 0 or a NaN, not 0.96.
Run the server with E3_CHECK=N (per-layer budget) and E3_CHECK_MIN_T above the largest CUDA-graph size, so the
checked calls are eager. Zero-norm references (the dummy profile run) do not consume the budget.
"""
import json
import os
import re
import sys
import urllib.request

MIN_COS = float(os.environ.get("MIN_COS", "0.999"))
LINE = re.compile(r"E3 check layer=(\d+) T=(\d+) rows=(\d+) ok=(\d) cos=([-\d.]+) rel_diff=([\d.]+)")
IDENT = re.compile(r"id_cos=([-\d.]+) prod_frac=([\d.]+)")  # identity check (added 2026-11-21; required)


def send(url, model, gsm, n):
    rows = [json.loads(line) for line in open(gsm)]
    for k in range(n):
        shots = rows[300 + 24 * k: 300 + 24 * (k + 1)]
        text = "\n\n".join(f"Question: {r['question']}\nAnswer: {r['answer']}" for r in shots[:-1])
        text += f"\n\nQuestion: {shots[-1]['question']}\nAnswer:"
        body = {"model": model, "prompt": text, "max_tokens": 8, "temperature": 0}
        req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.loads(r.read())
        print(f"prompt {k}: {out['usage']['prompt_tokens']} tokens -> {out['choices'][0]['text']!r}", flush=True)


def parse(paths, rowmap=None):
    expect = None
    if rowmap:
        rm = json.load(open(rowmap))
        expect = sorted(int(k) for k, v in rm["layers"].items() if v.get("warm"))
    ok_all = True
    summary = {}
    for rank, path in enumerate(paths):
        per, ident = {}, {}
        for line in open(path, errors="replace"):
            m = LINE.search(line)
            if m:
                layer, cos, ok = int(m.group(1)), float(m.group(5)), int(m.group(4))
                per.setdefault(layer, []).append((cos, ok, int(m.group(3)), float(m.group(6))))
                mi = IDENT.search(line)
                ident.setdefault(layer, []).append((float(mi.group(1)), float(mi.group(2))) if mi else (None, None))
        layers = sorted(per)
        mins = {k: min(c for c, _, _, _ in v) for k, v in per.items()}
        missing = [k for k in (expect or []) if k not in per]
        bad = {k: v for k, v in mins.items() if v < MIN_COS}
        notok = sum(1 for v in per.values() for _, o, _, _ in v if not o)
        n = sum(len(v) for v in per.values())
        # identity: every check must carry id_cos >= MIN_COS and prod_frac < 0.9999 (warm experts really masked)
        id_bad = sorted(k for k, v in ident.items() if any(c is None or c < MIN_COS or f >= 0.9999 for c, f in v))
        rank_ok = not missing and not bad and not id_bad and notok == 0 and n > 0
        ok_all &= rank_ok
        summary[f"rank{rank}"] = {
            "checks": n, "layers_checked": len(layers), "missing_layers": missing, f"below_{MIN_COS}": bad,
            "timeouts_in_checks": notok, "identity_fail_layers": id_bad,
            "cos_min": round(min(mins.values()), 6) if mins else None,
            "cos_mean": round(sum(c for v in per.values() for c, _, _, _ in v) / max(n, 1), 6),
            "worst_layer": min(mins, key=mins.get) if mins else None, "pass": rank_ok}
    summary["PASS"] = ok_all
    print(json.dumps(summary, indent=1))
    return ok_all


if __name__ == "__main__":
    if sys.argv[1] == "send":
        send(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 6)
    else:
        args = sys.argv[2:]
        rowmap = args.pop() if args and args[-1].endswith(".json") else None
        sys.exit(0 if parse(args, rowmap) else 1)
