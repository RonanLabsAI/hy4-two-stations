#!/usr/bin/env python3
"""Freeze every L6-4h input as token IDs, once, on the desk host (no GPU, no server).

  uv run --no-project --with tokenizers==0.23.3 --with jinja2==3.1.6 --with pyarrow==26.0.1 \
      python -I tools/hy4/build_inputs.py

Reads only files under tools/hy4/data/ (pinned tokenizer, chat template, raw datasets whose SHA-256 are
checked below). Writes tools/hy4/data/frozen/*.json and frozen/MANIFEST.json. Deterministic: rerunning
gives byte-identical files.

Why token IDs: llama-server and vLLM tokenize text differently at the edges (BOS insertion, special-token
parsing). Every rung therefore receives the same ID sequences and the harness gates on
usage.prompt_tokens == len(ids). Text is kept alongside only for scoring and inspection.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

import jinja2
import pyarrow.parquet as pq
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE/'data'
RAW = DATA/'raw'
OUT = DATA/'frozen'
CKPT = DATA/'hy4-fp8-d063b585'

RAW_SHA = {
    'gsm8k-test.parquet': 'ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59',
    'mmlu-pro-test.parquet': '0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8',
    'wikitext2-test.parquet': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91',
    'humaneval-test.parquet': '2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6',
}
CKPT_SHA = {
    'tokenizer.json': '2e3f3d22065c2be5765ce99ab2410411ae338b32ce809ea20dcdb61d00198e83',
    'chat_template.jinja': '11b12531403da5726b7e1f6003f479043b05999e6746d39af949cbcdfd715c35',
}
MMLU_SEED = 20261120          # L6-2's MMLU-Pro seed is still TBD; if L6-2 freezes first, adopt its seed+draw
MMLU_N = 500
GSM_N = 500                   # L6-2 Q1: first 500 GSM8K test problems
THINK_N = 40                  # thinking-on subset: first 40 of each frozen set
NEEDLE_LENGTHS = (32768, 131072)
NEEDLE_SEEDS = (11, 12, 13)
PREFILL_LENGTHS = (4096, 32768)
DRAFT_LENGTHS = (2048, 16384, 131072, 524288)   # prereg D: DFly2 acceptance and ms/verify step vs context
DRAFT_INSTRUCTION = '\n\nSummarize the document above in detail, section by section.'
LP_WINDOWS = 24               # per corpus kind (text, code)
LP_WINDOW_TOKENS = 1024
LP_FIRST_PROBE = 64
LP_STRIDE = 16

GSM_INSTRUCTION = ('Solve the following math problem. Show your reasoning briefly, then give the final answer '
                   'on the last line in the form "#### <number>".\n\n')
MMLU_INSTRUCTION = ('Answer the following multiple choice question. Think step by step if needed, then finish '
                    'with "The answer is (X)" where X is the letter of the correct option.\n\n')
ALIGN_TEXT = 'Alignment probe: 3.14159, naïve café, 東京, def f(x):\n    return x**2  # ok'
SPECIALS = ['<｜hy_start:opensource｜>', '<｜hy_middle:opensource｜>', '<｜hy_end:opensource｜>',
            '<｜reasoning_mode:opensource｜>', '<think:opensource>', '</think:opensource>']
NEEDLE_QUESTION = ('\n\nList every secret code number and its value that appears in the document above. '
                   'Answer only with the codes, one per line, as "code N: VALUE".')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_inputs():
    for name, want in RAW_SHA.items():
        got = sha(RAW/name)
        if got != want:
            sys.exit(f'{name}: sha256 {got} != frozen {want}')
    for name, want in CKPT_SHA.items():
        got = sha(CKPT/name)
        if got != want:
            sys.exit(f'{name}: sha256 {got} != frozen {want}')


class Renderer:
    def __init__(self):
        env = jinja2.Environment()
        env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
        self.t = env.from_string((CKPT/'chat_template.jinja').read_text())
        self.tok = Tokenizer.from_file(str(CKPT/'tokenizer.json'))

    def text(self, user, effort):
        return self.t.render(messages=[{'role': 'user', 'content': user}], add_generation_prompt=True,
                             reasoning_effort=effort)

    def ids(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def prompt(self, user, effort):
        text = self.text(user, effort)
        return dict(text=text, ids=self.ids(text))


def gsm8k(r):
    rows = pq.read_table(RAW/'gsm8k-test.parquet').to_pylist()[:GSM_N]
    items = []
    for i, row in enumerate(rows):
        gold = row['answer'].split('####')[-1].strip().replace(',', '')
        items.append(dict(id=f'gsm8k-test-{i:04d}', gold=gold, question=row['question'],
                          no_think=r.prompt(GSM_INSTRUCTION+row['question'], 'no_think')['ids'],
                          high=r.prompt(GSM_INSTRUCTION+row['question'], 'high')['ids'] if i < THINK_N else None))
    return items


def mmlu_subset():
    rows = pq.read_table(RAW/'mmlu-pro-test.parquet').to_pylist()
    by = {}
    for row in rows:
        by.setdefault(row['category'], []).append(row)
    rng = random.Random(MMLU_SEED)
    cats = sorted(by)
    total = len(rows)
    quota = {c: int(MMLU_N*len(by[c])/total) for c in cats}
    rem = MMLU_N-sum(quota.values())
    # largest remainders, ties by category name
    for c in sorted(cats, key=lambda c: (-(MMLU_N*len(by[c])/total-quota[c]), c))[:rem]:
        quota[c] += 1
    picked = []
    for c in cats:
        pool = sorted(by[c], key=lambda x: x['question_id'])
        picked += rng.sample(pool, quota[c])
    rng.shuffle(picked)
    return picked, quota


def mmlu(r):
    picked, quota = mmlu_subset()
    items = []
    for i, row in enumerate(picked):
        letters = 'ABCDEFGHIJ'[:len(row['options'])]
        body = row['question'].strip()+'\n\n'+'\n'.join(f'({l}) {o}' for l, o in zip(letters, row['options']))
        items.append(dict(id=f"mmlu-pro-{row['question_id']}", gold=row['answer'], category=row['category'],
                          n_options=len(letters), no_think=r.prompt(MMLU_INSTRUCTION+body, 'no_think')['ids'],
                          high=r.prompt(MMLU_INSTRUCTION+body, 'high')['ids'] if i < THINK_N else None))
    return items, quota


WORDS = ('river stone lantern harbor meadow copper signal winter orchard canyon ledger compass velvet beacon '
         'thunder garden quarry falcon timber marble cobalt prairie anchor violet summit glacier mirror cedar '
         'harvest saddle ember tundra ferry pebble willow cipher').split()


def filler(rng, n_words):
    out = []
    for i in range(n_words):
        out.append(rng.choice(WORDS))
        if i % 14 == 13:
            out[-1] += '.'
    return ' '.join(out)


def sized_prompt(r, build, target, effort='no_think'):
    """Grow/shrink filler until the rendered prompt is exactly `target` tokens (trim filler words)."""
    lo, hi = 1, target
    best = None
    while lo <= hi:
        mid = (lo+hi)//2
        text = r.text(build(mid), effort)
        n = len(r.ids(text))
        if n <= target:
            best = (mid, n); lo = mid+1
        else:
            hi = mid-1
    words, n = best
    return r.prompt(build(words), effort), n


def needles(r):
    items = []
    for L in NEEDLE_LENGTHS:
        for seed in NEEDLE_SEEDS:
            rng = random.Random(seed*1000+L)
            codes = [''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ23456789', k=8)) for _ in range(3)]
            base_rng_state = rng.getstate()

            def build(n_words, codes=codes, state=base_rng_state):
                g = random.Random(); g.setstate(state)
                words = filler(g, n_words).split(' ')
                for i, c in enumerate(codes):
                    pos = int(len(words)*(i+1)/4)
                    words.insert(pos, f'The secret code number {i+1} is {c}.')
                return ' '.join(words)+NEEDLE_QUESTION
            p, n = sized_prompt(r, build, L-256)
            items.append(dict(id=f'needle-{L//1024}k-s{seed}', length=L, prompt_tokens=n, codes=codes,
                              depths=[.25, .5, .75], ids=p['ids']))
    return items


def prefill_docs(r):
    items = []
    for L in PREFILL_LENGTHS:
        rng = random.Random(L)
        state = rng.getstate()

        def build(n_words, state=state):
            g = random.Random(); g.setstate(state)
            return filler(g, n_words)+'\n\nSummarize the text above in one sentence.'
        p, n = sized_prompt(r, build, L)
        items.append(dict(id=f'prefill-{L//1024}k', length=L, prompt_tokens=n, ids=p['ids']))
    # Nonce pool: single-token words; the harness overwrites 8 IDs after the system header with a fresh
    # random draw per request so no two prefill requests share a prefix beyond the header.
    pool = sorted({i for w in WORDS for i in r.ids(' '+w) if len(r.ids(' '+w)) == 1})
    return items, pool


def drafter_docs(r):
    """Prereg D prompts: wikitext-2-raw test text (natural prose, so draft acceptance is meaningful), tiled to
    reach the length, then a summarize instruction; thinking off; sized to exactly L-256 tokens so a 256-token
    reply fits in L. Added 2026-11-21 pre-freeze (change 5); the other frozen files are unchanged."""
    rows = pq.read_table(RAW/'wikitext2-test.parquet').to_pylist()
    words = ''.join(x['text'] for x in rows).split()
    items = []
    for L in DRAFT_LENGTHS:
        def build(n_words):
            reps = -(-n_words//len(words))
            return ' '.join((words*reps)[:n_words])+DRAFT_INSTRUCTION
        p, n = sized_prompt(r, build, L-256)
        items.append(dict(id=f'draft-{L//1024}k', length=L, prompt_tokens=n, ids=p['ids']))
    return items


def speed_prompts(r):
    corpus = json.loads((DATA/'study-c-reference-corpus.json').read_text())
    return [dict(id=p['id'], ids=r.prompt(p['prompt'], 'no_think')['ids']) for p in corpus['prompts']]


def speed_heldout_prompts(r):
    """Follow-up 1 (prereg change 8): 16 speed prompts outside the COUNT traffic that chose the warm experts.
    8 encyclopedia requests on wikitext-2 test article titles (every 4th top-level heading) and 8 HumanEval
    completions (tasks 120-127). COUNT saw only speed.json and the first 100 Q1/Q2 items."""
    rows = pq.read_table(RAW/'wikitext2-test.parquet').to_pylist()
    titles = [m.group(1).strip() for x in rows for m in [re.match(r'^ = ([^=].*?) = \n$', x['text'])] if m]
    titles = [t for t in titles if re.fullmatch(r"[A-Z][A-Za-z0-9 ,'().-]{2,60}", t)]
    he = pq.read_table(RAW/'humaneval-test.parquet').to_pylist()
    out = [dict(id=f'ho-wiki-{i:02d}', prompt=f'Write a detailed encyclopedia article about {t}.')
           for i, t in enumerate(titles[::4][:8])]
    out += [dict(id=f'ho-code-{i:02d}', prompt='Complete the following Python function and explain how it works.\n\n'+x['prompt'])
            for i, x in enumerate(he[120:128])]
    assert len(out) == 16, len(out)
    return [dict(id=p['id'], text=p['prompt'], ids=r.prompt(p['prompt'], 'no_think')['ids']) for p in out]


def logprob_corpus(r):
    """Windows of exactly LP_WINDOW_TOKENS raw-text tokens (no chat template), probed every LP_STRIDE tokens.

    Text: wikitext-2-raw test articles in order. Code: HumanEval prompt+canonical solution in task order.
    A probe at position p sends ids[:p] and records the top-k distribution of the next token; the true
    next token ids[p] is stored for NLL. Positions are identical for every engine by construction."""
    rows = pq.read_table(RAW/'wikitext2-test.parquet').to_pylist()
    text = ''.join(x['text'] for x in rows)
    he = pq.read_table(RAW/'humaneval-test.parquet').to_pylist()
    code = '\n\n'.join(x['prompt']+x['canonical_solution'] for x in he)
    out = []
    for kind, src in (('text', text), ('code', code)):
        ids = r.ids(src)
        need = LP_WINDOWS*LP_WINDOW_TOKENS
        if len(ids) < need:
            sys.exit(f'{kind} corpus too short: {len(ids)} < {need}')
        for w in range(LP_WINDOWS):
            win = ids[w*LP_WINDOW_TOKENS:(w+1)*LP_WINDOW_TOKENS]
            probes = list(range(LP_FIRST_PROBE, LP_WINDOW_TOKENS, LP_STRIDE))
            out.append(dict(id=f'lp-{kind}-{w:02d}', kind=kind, ids=win, probes=probes,
                            targets=[win[p] for p in probes]))
    return out


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, separators=(',', ':'), sort_keys=True)+'\n'
    (OUT/name).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    check_inputs()
    r = Renderer()
    manifest = {}
    templates = {e: r.text('{USER}', e) for e in ('no_think', 'high')}
    manifest['templates.json'] = dump('templates.json', dict(
        rendered=templates, ids={e: r.ids(t) for e, t in templates.items()},
        user_offset={e: len(r.ids(t.split('{USER}')[0])) for e, t in templates.items()},
        alignment_probe=dict(text=ALIGN_TEXT, ids=r.ids(ALIGN_TEXT)),
        specials={t: r.ids(t) for t in SPECIALS},
        think_open_id=r.ids('<think:opensource>')[0], think_close_id=r.ids('</think:opensource>')[0],
        eos_id=r.ids('<｜hy_end:opensource｜>')[0]))
    manifest['speed.json'] = dump('speed.json', speed_prompts(r))
    manifest['gsm8k.json'] = dump('gsm8k.json', gsm8k(r))
    items, quota = mmlu(r)
    manifest['mmlu_pro.json'] = dump('mmlu_pro.json', dict(seed=MMLU_SEED, quota=quota, items=items))
    manifest['needle.json'] = dump('needle.json', needles(r))
    docs, pool = prefill_docs(r)
    manifest['prefill.json'] = dump('prefill.json', dict(docs=docs, nonce_pool=pool, nonce_len=8))
    manifest['logprob_corpus.json'] = dump('logprob_corpus.json', logprob_corpus(r))
    manifest['drafter.json'] = dump('drafter.json', drafter_docs(r))
    manifest['speed_heldout.json'] = dump('speed_heldout.json', speed_heldout_prompts(r))
    manifest['_inputs'] = {**{f'raw/{k}': v for k, v in RAW_SHA.items()}, **{f'ckpt/{k}': v for k, v in CKPT_SHA.items()},
                           'study-c-reference-corpus.json': sha(DATA/'study-c-reference-corpus.json')}
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest, indent=1, sort_keys=True)+'\n')
    print(json.dumps(manifest, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
