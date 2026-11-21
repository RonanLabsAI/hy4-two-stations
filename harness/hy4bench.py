#!/usr/bin/env python3
"""L6-4h benchmark + quality harness for any OpenAI-compatible server (llama-server or vLLM). Stdlib only.

Every prompt is sent as frozen token IDs (data/frozen/, built by build_inputs.py) to /v1/completions, so
all rungs see identical token sequences. Gates are per request and are recorded, never widened:
  G1 usage present and usage.prompt_tokens == len(ids)          (token counts only from server usage)
  G2 thinking off: no </think> token text in the reply
  G3 decode cells: completion_tokens == max_tokens (ignore_eos honoured)
Concurrency: N worker threads each send their requests; aggregate = sum completion tokens / (last end -
first start). C1 decode rate uses streaming only for timing the first token; tokens come from usage.

Subcommands (each writes one JSONL file of raw records + one summary JSON under --out):
  check      health, model id, engine flavour, prompt-token gate on 3 prompts, tokenization alignment
  speed      C1 (16 prompts one at a time) and C16 (16 at once) decode, 512 tokens, ignore_eos, T=0
  prefill    cold prefill at 4K and 32K (unique nonce per request), TTFT and prompt_tokens/TTFT
  needle     exact-match retrieval of 3 codes at 32K and 128K (3 docs each)
  quality    GSM8K-500 and MMLU-Pro-500 thinking off; --think adds the 40+40 thinking-on subset
  logprobs   top-k next-token capture on the frozen text+code corpus (identical positions for every rung)
  all        check, speed, prefill, needle, quality, logprobs in that order

Example:
  python3 -I tools/hy4/hy4bench.py all --base http://127.0.0.1:18080 --engine llamacpp --rung stq1_0 \
      --out runs/l6-4h/stq1_0-left-$(date +%Y%m%d-%H%M)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = HERE/'data'/'frozen'
VERSION = '0.1.0'


def load(name):
    return json.loads((FROZEN/name).read_text())


# ---------------------------------------------------------------- HTTP client

class Client:
    def __init__(self, base, engine, model=None, timeout=7200):
        self.base = base.rstrip('/')
        self.engine = engine
        self.timeout = timeout
        self.model = model or self.models()[0]

    def _req(self, path, body=None, timeout=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base+path, data=data, headers={'Content-Type': 'application/json'})
        return urllib.request.urlopen(req, timeout=timeout or self.timeout)

    def get(self, path):
        with self._req(path, timeout=30) as r:
            return json.loads(r.read() or b'{}')

    def post(self, path, body):
        with self._req(path, body) as r:
            return json.loads(r.read())

    def models(self):
        return [m['id'] for m in self.get('/v1/models').get('data', [])]

    def completion(self, ids, max_tokens, ignore_eos=False, logprobs=None, stream=False, seed=1234):
        """/v1/completions with a token-ID prompt, greedy. Returns a normalised record."""
        body = {'model': self.model, 'prompt': ids, 'max_tokens': max_tokens, 'temperature': 0.0, 'top_p': 1.0,
                'seed': seed, 'stream': stream}
        if self.engine == 'vllm':  # keep think tags in the text, as llama-server -sp does
            body['skip_special_tokens'] = False
            body['spaces_between_special_tokens'] = False
        if ignore_eos:
            body['ignore_eos'] = True
        if stream:
            body['stream_options'] = {'include_usage': True}
        if logprobs:
            body['logprobs'] = logprobs
            if self.engine == 'vllm':
                body['return_tokens_as_token_ids'] = True
        t0 = time.monotonic()
        if not stream:
            resp = self.post('/v1/completions', body)
            t1 = time.monotonic()
            ch = resp['choices'][0]
            return dict(t0=t0, t_end=t1, ttft=None, text=ch.get('text', ''), finish=ch.get('finish_reason'),
                        usage=resp.get('usage'), logprobs=ch.get('logprobs'), timings=resp.get('timings'))
        text, usage, t_first, finish, timings = [], None, None, None, None
        with self._req('/v1/completions', body) as r:
            for raw in r:
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                ev = json.loads(data)
                if ev.get('usage'):
                    usage = ev['usage']
                if ev.get('timings'):
                    timings = ev['timings']
                for ch in ev.get('choices') or []:
                    if ch.get('text'):
                        if t_first is None:
                            t_first = time.monotonic()
                        text.append(ch['text'])
                    if ch.get('finish_reason'):
                        finish = ch['finish_reason']
        t1 = time.monotonic()
        return dict(t0=t0, t_end=t1, ttft=(t_first or t1)-t0, t_first=t_first, text=''.join(text), finish=finish,
                    usage=usage, logprobs=None, timings=timings)

    def topk(self, ids, k):
        """Next-token top-k after prompt `ids` (generated-token logprobs of a 1-token completion).

        llama-server: /completion n_probs (pre-sampling softmax of raw logits; no prompt logprobs, echo is
        rejected). vLLM: /v1/completions logprobs=k with return_tokens_as_token_ids (raw_logprobs mode;
        server must run with --max-logprobs >= k). Returns ({token_id: logprob}, sampled_id, usage)."""
        if self.engine == 'llamacpp':
            body = {'prompt': ids, 'n_predict': 1, 'n_probs': k, 'temperature': 0.0, 'cache_prompt': True,
                    'post_sampling_probs': False, 'id_slot': 0, 'return_tokens': True}
            resp = self.post('/completion', body)
            eog_retry = False
            if not resp.get('completion_probabilities'):
                # Possibly an end-of-generation greedy token: retry with ignore_eos (biases sampling only; raw
                # n_probs unchanged). Sampled = raw argmax.
                resp = self.post('/completion', dict(body, ignore_eos=True))
                eog_retry = True
            usage = {'prompt_tokens': resp.get('tokens_evaluated'), 'completion_tokens': resp.get('tokens_predicted'),
                     'eog_retry': eog_retry}
            if not resp.get('completion_probabilities'):
                # Greedy token is an incomplete UTF-8 byte piece (e.g. half of ' ➞'): llama-server never stores
                # its probs, streamed or not (server-context.cpp add_token inside `if (!incomplete)`). Found at
                # probe 2761 (lp-code-22 pos 64) on STQ1_0 and Q4_K_M, 2026-11-20. Recorded as a gate failure
                # (no top-k); analyze.drift excludes it from both sides of every comparison and counts it.
                usage['no_probs'] = 'incomplete_utf8'
                toks = resp.get('tokens') or []
                return {}, (int(toks[0]) if toks else None), usage
            cp = resp['completion_probabilities'][0]
            top = {int(x['id']): float(x['logprob']) for x in cp['top_logprobs']}
            return top, (max(top, key=top.get) if eog_retry else int(cp['id'])), usage
        rec = self.completion(ids, 1, logprobs=k)
        lp = rec['logprobs']
        top = {int(t.split(':', 1)[1]): float(v) for t, v in lp['top_logprobs'][0].items()}
        sampled = int(lp['tokens'][0].split(':', 1)[1])
        return top, sampled, rec['usage']

    def tokenize(self, text):
        if self.engine == 'llamacpp':
            return self.post('/tokenize', {'content': text, 'add_special': False, 'parse_special': True})['tokens']
        return self.post('/tokenize', {'model': self.model, 'prompt': text, 'add_special_tokens': False})['tokens']

    def detokenize(self, ids):
        if self.engine == 'llamacpp':
            return self.post('/detokenize', {'tokens': ids})['content']
        return self.post('/detokenize', {'model': self.model, 'tokens': ids})['prompt']

    def metrics(self):
        try:
            with self._req('/metrics', timeout=30) as r:
                return r.read().decode()
        except Exception as e:  # metrics are secondary; absence is recorded, not fatal
            return f'# unavailable: {e}'


# ---------------------------------------------------------------- recording

class Recorder:
    def __init__(self, out, name):
        self.path = Path(out)/f'{name}.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():  # never mix two runs in one file; pick a fresh --out
            raise SystemExit(f'{self.path} exists; refusing to append')
        self.f = self.path.open('x')
        self.lock = threading.Lock()

    def write(self, rec):
        with self.lock:
            self.f.write(json.dumps(rec, sort_keys=True)+'\n'); self.f.flush()

    def close(self):
        self.f.close()


def gate_prompt(rec, ids):
    u = rec.get('usage') or {}
    return u.get('prompt_tokens') == len(ids)


# Special-token text differs by engine: HF/vLLM render '<think:opensource>', the AngelSlim GGUFs carry the same
# IDs under hashed names ('<think:6124c78e>', '<｜hy_end:6124c78e｜>'), found in the 2026-11-20 STQ1_0 check.
# So think tags are matched by pattern, and every special-token string is stripped before scoring (the hash
# digits would otherwise be read as a GSM8K answer). Servers must emit specials: llama-server -sp, vLLM
# skip_special_tokens=false.
THINK_CLOSE_RE = re.compile(r'</think:[^>\s]*>')
THINK_OPEN_RE = re.compile(r'<think:[^>\s]*>')
SPECIAL_RE = re.compile(r'<｜[^｜\s]*｜>|</?think:[^>\s]*>')


def has_think_close(text):
    return bool(THINK_CLOSE_RE.search(text))


def strip_specials(text):
    return SPECIAL_RE.sub('', text)


def gate_thinking_off(text):
    return not THINK_CLOSE_RE.search(text) and not THINK_OPEN_RE.search(text)


def run_parallel(fn, jobs, workers):
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, jobs))


# ---------------------------------------------------------------- measures

def cmd_check(c, out, args):
    tpl = load('templates.json')
    speed = load('speed.json')
    res = dict(engine=c.engine, model=c.model, base_is_loopback=bool(re.match(r'https?://(127\.|localhost)', c.base)))
    recs = []
    for p in speed[:3]:
        r = c.completion(p['ids'], 8)
        recs.append(dict(id=p['id'], prompt_gate=gate_prompt(r, p['ids']), usage=r['usage'], text=r['text']))
    res['prompt_gate'] = all(x['prompt_gate'] for x in recs)
    res['samples'] = recs
    # Tokenization alignment. The harness always sends frozen HF token IDs, so positions are identical by
    # construction; what must hold is that every ID means the same string to this server (detok_match, a stop
    # gate). tokenize_match (server text->IDs equals HF) is reported; a mismatch there is expected to matter
    # only for text prompts, which the harness never sends.
    ap = tpl['alignment_probe']
    try:
        res['detok_match'] = c.detokenize(ap['ids']) == ap['text']
        server_ids = c.tokenize(ap['text'])
        res['tokenize_match'] = server_ids == ap['ids']
        res['tokenize_server_ids'] = server_ids
        # every special ID in the frozen prompts must be one special token on this server: its server string
        # tokenizes back to exactly that ID (round trip). The string itself may differ by engine (GGUF hashed
        # names), so it is recorded, not compared.
        srv = {t: c.detokenize(ids) for t, ids in tpl['specials'].items()}
        res['specials_server'] = srv
        res['specials_match'] = all(c.tokenize(srv[t]) == ids for t, ids in tpl['specials'].items())
        res['detok_match'] = res['detok_match'] and res['specials_match']
    except Exception as e:
        res['detok_match'] = False
        res['tokenizer_error'] = repr(e)
    res['metrics_head'] = c.metrics()[:2000]
    write_summary(out, 'check', res)
    return res


def cmd_speed(c, out, args):
    prompts = load(args.speed_prompts)
    name = 'speed' if args.speed_prompts == 'speed.json' else 'speed-'+args.speed_prompts.removeprefix('speed_').removesuffix('.json')
    rec = Recorder(out, name)
    res = {}
    n_tok = args.decode_tokens
    for cell, conc in (('c1', 1), ('c16', 16)):
        rounds = []
        for rnd in range(args.warm+args.rounds):
            stream = conc == 1

            def one(p):
                try:
                    r = c.completion(p['ids'], n_tok, ignore_eos=True, stream=stream)
                except Exception as e:  # a failed request is a failed gate, recorded, never retried
                    row = dict(cell=cell, round=rnd, warm=rnd < args.warm, id=p['id'], completion_tokens=None,
                               t0=time.monotonic(), t_end=time.monotonic(), decode_tps=None, error=repr(e),
                               gates=dict(request=False))
                    rec.write(row)
                    return row
                ct = (r['usage'] or {}).get('completion_tokens')
                g = dict(prompt=gate_prompt(r, p['ids']), exact_len=ct == n_tok, thinking_off=gate_thinking_off(r['text']))
                decode = (ct-1)/(r['t_end']-r['t_first']) if stream and ct and r.get('t_first') and r['t_end'] > r['t_first'] else None
                row = dict(cell=cell, round=rnd, warm=rnd < args.warm, id=p['id'], completion_tokens=ct,
                           prompt_tokens=(r['usage'] or {}).get('prompt_tokens'), t0=r['t0'], t_end=r['t_end'],
                           ttft=r['ttft'], decode_tps=decode, gates=g, timings=r['timings'], finish=r['finish'])
                rec.write(row)
                return row
            t_start = time.monotonic()
            rows = run_parallel(one, prompts, conc)
            wall = max(x['t_end'] for x in rows)-min(x['t0'] for x in rows)
            toks = sum(x['completion_tokens'] or 0 for x in rows)
            ok = all(all(x['gates'].values()) for x in rows)
            summary = dict(round=rnd, warm=rnd < args.warm, completion_tokens=toks, wall_s=wall,
                           throughput_tps=toks/wall if wall > 0 else None, gates_ok=ok, elapsed=time.monotonic()-t_start)
            if conc == 1:
                d = [x['decode_tps'] for x in rows if x['decode_tps']]
                summary['mean_decode_tps_per_reply'] = sum(d)/len(d) if d else None
                # C1 throughput is per-request sequential: tokens / sum of request walls
                summary['throughput_tps'] = toks/sum(x['t_end']-x['t0'] for x in rows)
            rounds.append(summary)
        kept = [r for r in rounds if not r['warm']]
        vals = [r['throughput_tps'] for r in kept if r['throughput_tps']]
        res[cell] = dict(rounds=rounds, mean_tps=sum(vals)/len(vals) if vals else None,
                         cv=cv(vals), all_gates_ok=all(r['gates_ok'] for r in kept))
    rec.close()
    write_summary(out, name, res)
    return res


def cv(vals):
    if len(vals) < 2:
        return None
    mu = sum(vals)/len(vals)
    sd = (sum((v-mu)**2 for v in vals)/(len(vals)-1))**.5
    return sd/mu


def nonce_ids(base_ids, pool, n, rng, header_len):
    ids = list(base_ids)
    for i in range(n):
        ids[header_len+i] = rng.choice(pool)
    return ids


def cmd_prefill(c, out, args):
    d = load('prefill.json'); tpl = load('templates.json')
    header = tpl['user_offset']['no_think']  # first token of the user content
    rng = random.Random(time.time_ns())
    rec = Recorder(out, 'prefill')
    res = {}
    for doc in d['docs']:
        rows = []
        for rep in range(args.prefill_reps):
            ids = nonce_ids(doc['ids'], d['nonce_pool'], d['nonce_len'], rng, header)
            r = c.completion(ids, 1, stream=True)
            pt = (r['usage'] or {}).get('prompt_tokens')
            cached = ((r['usage'] or {}).get('prompt_tokens_details') or {}).get('cached_tokens')
            # cold gate: the cache may reuse the shared template header and, by chance, a few leading nonce
            # tokens, never past the nonce (None = engine did not report cached tokens)
            cold = cached is None or cached < header+d['nonce_len']
            row = dict(id=doc['id'], rep=rep, prompt_tokens=pt, cached_tokens=cached, ttft=r['ttft'],
                       prefill_tps=pt/r['ttft'] if pt and r['ttft'] else None, gate_prompt=gate_prompt(r, ids),
                       gate_cold=cold, timings=r['timings'], nonce=ids[header:header+d['nonce_len']])
            rec.write(row); rows.append(row)
        v = sorted(x['prefill_tps'] for x in rows if x['prefill_tps'])
        res[doc['id']] = dict(median_prefill_tps=v[len(v)//2] if v else None, reps=len(rows),
                              gates_ok=all(x['gate_prompt'] and x['gate_cold'] for x in rows))
    rec.close()
    write_summary(out, 'prefill', res)
    return res


def score_needle(text, codes):
    """Exact match: each code must appear as a whole token (no adjacent [A-Z0-9])."""
    return [re.search(rf'(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])', text) is not None for code in codes]


def cmd_needle(c, out, args):
    items = load('needle.json')
    rec = Recorder(out, 'needle')
    res = {}
    for it in items:
        if args.max_needle and it['length'] > args.max_needle:
            res[it['id']] = dict(skipped=f"length {it['length']} > --max-needle {args.max_needle}")
            continue
        r = c.completion(it['ids'], 96)
        hits = score_needle(r['text'], it['codes'])
        row = dict(id=it['id'], length=it['length'], hits=hits, all_found=all(hits), text=r['text'],
                   prompt_tokens=(r['usage'] or {}).get('prompt_tokens'), gate_prompt=gate_prompt(r, it['ids']),
                   thinking_off=gate_thinking_off(r['text']), wall=r['t_end']-r['t0'])
        rec.write(row)
        res[it['id']] = dict(all_found=row['all_found'], hits=hits, gates_ok=row['gate_prompt'] and row['thinking_off'])
    rec.close()
    write_summary(out, 'needle', res)
    return res


NUM_RE = re.compile(r'-?\d[\d,]*(?:\.\d+)?')


def score_gsm8k(text, gold):
    """Study D rule: number after '####', else the last number; exact match after removing commas."""
    tail = text.split('####')[-1] if '####' in text else text
    nums = NUM_RE.findall(tail)
    if not nums:
        return False, None
    pred = (nums[0] if '####' in text else nums[-1]).replace(',', '').rstrip('.')
    try:
        return float(pred) == float(gold), pred
    except ValueError:
        return False, pred


MMLU_RE = re.compile(r'answer is \(?([A-J])\)?', re.I)
MMLU_FALLBACK = re.compile(r'\(([A-J])\)')


def score_mmlu(text, gold):
    """Frozen: last 'answer is (X)' match; else the last '(X)'; else wrong."""
    m = MMLU_RE.findall(text)
    pred = m[-1].upper() if m else None
    if pred is None:
        f = MMLU_FALLBACK.findall(text)
        pred = f[-1] if f else None
    return pred == gold, pred


def answer_text(text):
    """Thinking-on replies: score only the part after the last </think> (content), like a reasoning parser."""
    return strip_specials(THINK_CLOSE_RE.split(text)[-1])


def cmd_quality(c, out, args):
    sets = [('gsm8k', load('gsm8k.json'), score_gsm8k, 1024), ('mmlu_pro', load('mmlu_pro.json')['items'], score_mmlu, 1024)]
    rec = Recorder(out, 'quality')
    res = {}
    modes = ['no_think']+(['high'] if args.think else [])
    for mode in modes:
        for name, items, scorer, cap in sets:
            cap = args.think_cap if mode == 'high' else cap
            todo = [x for x in items if x[mode] is not None]
            if args.limit:
                todo = todo[:args.limit]

            def one(it):
                try:
                    r = c.completion(it[mode], cap)
                except Exception as e:  # recorded as wrong and as a gate failure
                    row = dict(set=name, mode=mode, id=it['id'], correct=False, pred=None, gold=it['gold'],
                               completion_tokens=None, hit_cap=False, empty=True, gate_prompt=False,
                               thinking_gate=False, error=repr(e), text='')
                    rec.write(row)
                    return row
                text = strip_specials(r['text']) if mode == 'no_think' else answer_text(r['text'])
                ok, pred = scorer(text, it['gold'])
                ct = (r['usage'] or {}).get('completion_tokens')
                row = dict(set=name, mode=mode, id=it['id'], correct=ok, pred=pred, gold=it['gold'], completion_tokens=ct,
                           hit_cap=ct == cap, empty=not text.strip(), gate_prompt=gate_prompt(r, it[mode]),
                           thinking_gate=gate_thinking_off(r['text']) if mode == 'no_think' else (has_think_close(r['text']) or ct == cap),
                           text=r['text'])
                rec.write(row)
                return row
            rows = run_parallel(one, todo, args.think_inflight if mode == 'high' else args.inflight)
            n = len(rows); k = sum(r['correct'] for r in rows)
            res[f'{name}:{mode}'] = dict(n=n, correct=k, accuracy=k/n if n else None, wilson95=wilson(k, n),
                                         cap_hits=sum(r['hit_cap'] for r in rows), empty=sum(r['empty'] for r in rows),
                                         gate_failures=sum(not (r['gate_prompt'] and r['thinking_gate']) for r in rows))
    rec.close()
    write_summary(out, 'quality', res)
    return res


def wilson(k, n, z=1.959963984540054):
    if not n:
        return None
    p = k/n; d = 1+z*z/n
    c = p+z*z/(2*n); h = z*((p*(1-p)/n+z*z/(4*n*n))**.5)
    return [(c-h)/d, (c+h)/d]


def cmd_logprobs(c, out, args):
    corpus = load('logprob_corpus.json')
    if args.limit:
        corpus = corpus[:args.limit]
    rec = Recorder(out, 'logprobs')
    n = bad = nop = 0
    for win in corpus:
        for p, target in zip(win['probes'], win['targets']):
            ids = win['ids'][:p]
            top, sampled, usage = c.topk(ids, args.topk)
            # both engines report the full prompt length (vLLM usage.prompt_tokens; llama-server tokens_evaluated)
            ok = len(top) >= args.topk and (usage or {}).get('prompt_tokens') == len(ids)
            rec.write(dict(window=win['id'], kind=win['kind'], pos=p, target=target, sampled=sampled,
                           top=sorted(top.items(), key=lambda kv: -kv[1]), gate=ok,
                           eog_retry=(usage or {}).get('eog_retry', False), no_probs=(usage or {}).get('no_probs')))
            n += 1; bad += not ok; nop += bool((usage or {}).get('no_probs'))
    rec.close()
    res = dict(probes=n, gate_failures=bad, no_probs_incomplete_utf8=nop, topk=args.topk, corpus_sha256=hashlib.sha256((FROZEN/'logprob_corpus.json').read_bytes()).hexdigest())
    write_summary(out, 'logprobs', res)
    return res


SPEC_KEYS = ('vllm:spec_decode_num_drafts_total', 'vllm:spec_decode_num_draft_tokens_total',
             'vllm:spec_decode_num_accepted_tokens_total')


def spec_counters(c):
    """Sum vLLM spec-decode counters over label sets. None when the server exposes none of them."""
    v, seen = dict.fromkeys(SPEC_KEYS, 0.0), False
    for line in c.metrics().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        name = line.split('{', 1)[0].split(' ', 1)[0]
        if name in v:
            v[name] += float(line.rsplit(' ', 1)[1]); seen = True
    return v if seen else None


def cmd_drafter(c, out, args):
    """Prereg D: DFly2 acceptance and ms per verify step at C1 vs context (2K/16K/128K/512K), thinking off,
    --draft-tokens output tokens (ignore_eos), --draft-reps reps. Counters are /metrics deltas around each request
    (nothing else may run on the server). ms/step = decode wall after the first token / verify steps."""
    docs = [d for d in load('drafter.json') if not args.max_draft or d['length'] <= args.max_draft]
    rec = Recorder(out, 'drafter')
    res = {}
    for d in docs:
        rows = []
        for rep in range(args.draft_reps):
            m0 = spec_counters(c)
            r = c.completion(d['ids'], args.draft_tokens, ignore_eos=True, stream=True)
            m1 = spec_counters(c)
            dd = {k.split('num_')[1].replace('_total', ''): (m1[k]-m0[k]) for k in SPEC_KEYS} if m0 and m1 else {}
            steps, drafted, acc = dd.get('drafts'), dd.get('draft_tokens'), dd.get('accepted_tokens')
            ct = (r['usage'] or {}).get('completion_tokens')
            dec = (r['t_end']-r['t_first']) if r['t_first'] else None
            row = dict(id=d['id'], length=d['length'], rep=rep, prompt_tokens=(r['usage'] or {}).get('prompt_tokens'),
                       completion_tokens=ct, ttft=r['ttft'], decode_s=dec, verify_steps=steps, draft_tokens=drafted,
                       accepted_tokens=acc, acceptance=(acc/drafted if drafted else None),
                       mean_accepted_len=(1+acc/steps if steps else None),
                       ms_per_step=(1000*dec/steps if steps and dec else None),
                       decode_tps=((ct-1)/dec if ct and dec else None),
                       gate_prompt=gate_prompt(r, d['ids']), gate_len=ct == args.draft_tokens, counters=m0 is not None)
            rec.write(row); rows.append(row)
        ok = [x for x in rows if x['gate_prompt'] and x['gate_len'] and x['ms_per_step']]
        med = lambda k: sorted(x[k] for x in ok)[len(ok)//2] if ok else None
        res[d['id']] = dict(length=d['length'], reps=len(rows), gates_ok=len(ok) == len(rows), ms_per_step_median=med('ms_per_step'),
                            acceptance_median=med('acceptance'), mean_accepted_len_median=med('mean_accepted_len'),
                            decode_tps_median=med('decode_tps'))
    rec.close()
    write_summary(out, 'drafter', res)
    return res


def write_summary(out, name, res):
    p = Path(out)/f'{name}.summary.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, sort_keys=True, default=str)+'\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['check', 'speed', 'prefill', 'needle', 'quality', 'logprobs', 'drafter', 'all'])
    ap.add_argument('--base', required=True)
    ap.add_argument('--engine', required=True, choices=['llamacpp', 'vllm'])
    ap.add_argument('--rung', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model')
    ap.add_argument('--warm', type=int, default=1)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--decode-tokens', type=int, default=512)
    ap.add_argument('--speed-prompts', default='speed.json', help='frozen prompt file for speed (speed_heldout.json = follow-up 1)')
    ap.add_argument('--prefill-reps', type=int, default=3)
    ap.add_argument('--max-needle', type=int, default=0, help='skip needle docs longer than this (0 = none)')
    ap.add_argument('--inflight', type=int, default=16)
    ap.add_argument('--think', action='store_true', help='also run the thinking-on subset')
    ap.add_argument('--think-cap', type=int, default=16384)
    ap.add_argument('--think-inflight', type=int, default=8, help='in-flight requests for thinking-on (KV budget)')
    ap.add_argument('--topk', type=int, default=64)
    ap.add_argument('--draft-reps', type=int, default=3)
    ap.add_argument('--draft-tokens', type=int, default=256)
    ap.add_argument('--max-draft', type=int, default=0, help='skip drafter docs longer than this (0 = none)')
    ap.add_argument('--limit', type=int, default=0, help='first N items/windows only (smoke)')
    args = ap.parse_args(argv)
    c = Client(args.base, args.engine, args.model)
    out = Path(args.out)
    meta = dict(harness_version=VERSION, harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                frozen_manifest=json.loads((FROZEN/'MANIFEST.json').read_text()), engine=args.engine, rung=args.rung,
                model=c.model, argv=sys.argv if argv is None else argv, started_unix=time.time())
    write_summary(out, 'meta', meta)
    cmds = ['check', 'speed', 'prefill', 'needle', 'quality', 'logprobs'] if args.cmd == 'all' else [args.cmd]
    if args.cmd == 'all' and args.engine == 'vllm':
        cmds.append('drafter')  # DFly2 is a vLLM-only feature here
    results = {}
    for name in cmds:
        results[name] = globals()[f'cmd_{name}'](c, out, args)
        if name == 'check' and not (results[name]['prompt_gate'] and results[name]['detok_match']):
            print('STOP: prompt-token or detokenization gate failed in check; nothing else runs', file=sys.stderr)
            return 2
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('samples', 'metrics_head', 'rounds')} if isinstance(v, dict) else v
                      for k, v in results.items()}, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
