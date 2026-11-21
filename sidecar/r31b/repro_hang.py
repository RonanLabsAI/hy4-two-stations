# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""repro_hang.py -- R3.1b reproducer + soak for the multi-node DSpark desync (R31B-DSPARK-HANG.md).

The R3.1 trigger is a MIXED step: new prompts prefill while other requests decode with K=5 drafts, so the target batch
holds more tokens than the drafter's FULL-graph context rows (num_reqs_padded x 6). DSpark then stores the excess
context K/V eagerly (_precompute_context_kv), a DeepGEMM fp8 GEMM with M = target tokens - graph rows. A NEW M range
means a new DeepGEMM kernel that is JIT-loaded in the hot path; at --gpu-memory-utilization 0.99 that load hit
CUDA_ERROR_OUT_OF_MEMORY on rank 1 -> rank 1 skipped the drafter -> cross-node collective desync -> hang.

Modes (all against the OpenAI chat endpoint; every request has a client timeout, so a hang is DETECTED, not waited on):
  arrival  the exact R3.1 arrival shape: 2 GSM8K thinking prompts, then 14 more ~2 s later (C16), x ROUNDS rounds
  sweep    15 long decodes (ignore_eos) keep the batch in spec decode while single prompts of SWEEP lengths (geometric,
           50..7000 tokens of synthetic text) arrive one by one, so the eager precompute sees ~30 different M values
  soak     DURATION seconds of mixed traffic at C16: GSM8K thinking/no-think (T0), synthetic random-id prompts
           (low acceptance), long prompts (4-12K), short chats; staggered arrivals; T>0 requests carry an explicit seed
Exit code: 0 = every request completed; 2 = any request timed out or the server stopped answering (= hang/desync);
           1 = HTTP errors without a hang.

usage: repro_hang.py <base http://host:8010> <model> <gsm8k_test.jsonl> <mode> <out.jsonl>
       env: ROUNDS=3 REQ_TIMEOUT=240 DURATION=1800 CONC=16 SEED=7
"""
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE, MODEL, GSM, MODE, OUT = sys.argv[1:6]
ROUNDS = int(os.environ.get("ROUNDS", "3"))
REQ_TIMEOUT = float(os.environ.get("REQ_TIMEOUT", "240"))
DURATION = float(os.environ.get("DURATION", "1800"))
CONC = int(os.environ.get("CONC", "16"))
rng = random.Random(int(os.environ.get("SEED", "7")))
rows = [json.loads(line)["question"] for line in open(GSM)]
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec "
         "romeo sierra tango uniform victor whiskey xray yankee zulu").split()
lock = threading.Lock()
stats = {"ok": 0, "timeout": 0, "http_error": 0, "other": 0}
outf = open(OUT, "a")


def synth(n_tokens: int, kind: str = "words") -> str:
    if kind == "ids":  # catid-like: random ids, near-zero draft acceptance
        return " ".join(str(rng.randrange(10**6)) for _ in range(max(1, n_tokens // 3)))
    return " ".join(rng.choice(WORDS) for _ in range(max(1, int(n_tokens * 0.9))))


def chat(tag: str, content: str, max_tokens: int, think: bool, temperature: float = 0.0, ignore_eos: bool = False):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": temperature, "chat_template_kwargs": {"thinking": think}}
    if think:
        body["chat_template_kwargs"]["reasoning_effort"] = "high"
    if temperature > 0:
        body["seed"] = rng.randrange(2**31)  # explicit seed: per-rank sampling streams cannot differ
    if ignore_eos:
        body["ignore_eos"] = True
    t0 = time.time()
    rec = {"tag": tag, "t0": round(t0, 3), "max_tokens": max_tokens, "think": think, "T": temperature}
    try:
        req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as r:
            d = json.load(r)
        rec.update(status="ok", ctoks=d["usage"]["completion_tokens"], ptoks=d["usage"]["prompt_tokens"],
                   finish=d["choices"][0]["finish_reason"])
        key = "ok"
    except urllib.error.HTTPError as e:
        rec.update(status=f"http {e.code}")
        key = "http_error"
    except TimeoutError:
        rec.update(status="timeout")
        key = "timeout"
    except Exception as e:  # socket timeouts surface as URLError/OSError
        timed_out = "timed out" in repr(e)
        rec.update(status="timeout" if timed_out else f"error {e!r}"[:200])
        key = "timeout" if timed_out else "other"
    rec["wall"] = round(time.time() - t0, 2)
    with lock:
        stats[key] += 1
        outf.write(json.dumps(rec) + "\n")
        outf.flush()
    return rec


def health() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=10):
            return True
    except Exception:
        return False


def mode_arrival():
    for rnd in range(ROUNDS):
        qs = rng.sample(rows, 16)
        with ThreadPoolExecutor(16) as ex:
            futs = [ex.submit(chat, f"arr{rnd}-{i}", qs[i], 1024, True) for i in range(2)]
            time.sleep(2.0)
            futs += [ex.submit(chat, f"arr{rnd}-{i}", qs[i], 1024, True) for i in range(2, 16)]
            [f.result() for f in futs]
        print(f"arrival round {rnd}: {stats}", flush=True)
        if stats["timeout"]:
            return


def sweep_lengths(n=30, lo=50, hi=7000):
    return sorted({int(round(lo * (hi / lo) ** (i / (n - 1)))) for i in range(n)})


def mode_sweep():
    lens = sweep_lengths()
    bg_tokens = 400 + 40 * len(lens)  # keep the 15 decodes alive for the whole sweep
    with ThreadPoolExecutor(16) as ex:
        bg = [ex.submit(chat, f"bg{i}", rng.choice(rows), bg_tokens, False, 0.0, True) for i in range(15)]
        time.sleep(8)
        for n in lens:
            chat(f"sweep{n}", synth(n), 16, False)
            if stats["timeout"]:
                break
        [f.result() for f in bg]
    print(f"sweep {len(lens)} lengths {lens[0]}..{lens[-1]}: {stats}", flush=True)


def soak_job(i: int):
    k = rng.random()
    if k < 0.35:
        return chat(f"soak{i}-gsm", rng.choice(rows), 4096, True)
    if k < 0.50:
        return chat(f"soak{i}-gsmnt", rng.choice(rows), 512, False)
    if k < 0.65:
        return chat(f"soak{i}-ids", synth(rng.randint(200, 6000), "ids"), rng.choice([256, 1024]), False, 0.0, True)
    if k < 0.75:
        return chat(f"soak{i}-long", synth(rng.randint(4000, 12000)), 256, False)
    if k < 0.90:
        return chat(f"soak{i}-chat", synth(rng.randint(10, 120)), rng.randint(16, 400), False, 0.6)
    return chat(f"soak{i}-gsmT", rng.choice(rows), 2048, True, 0.6)


def mode_soak():
    t_end = time.time() + DURATION
    i = 0
    with ThreadPoolExecutor(CONC) as ex:
        live = set()
        while time.time() < t_end and not stats["timeout"]:
            live = {f for f in live if not f.done()}
            while len(live) < CONC:
                live.add(ex.submit(soak_job, i))
                i += 1
                time.sleep(rng.uniform(0.0, 1.5))
            time.sleep(0.5)
            if i % 50 == 0:
                print(f"soak t={DURATION - (t_end - time.time()):.0f}s submitted {i}: {stats}", flush=True)
        [f.result() for f in live]
    print(f"soak done: submitted {i}: {stats}", flush=True)


if __name__ == "__main__":
    if not health():
        print("server not healthy at start")
        sys.exit(2)
    {"arrival": mode_arrival, "sweep": mode_sweep, "soak": mode_soak}[MODE]()
    alive = health()
    summary = {"mode": MODE, **stats, "healthy_after": alive}
    print(json.dumps(summary), flush=True)
    sys.exit(2 if stats["timeout"] or not alive else (1 if stats["http_error"] or stats["other"] else 0))
