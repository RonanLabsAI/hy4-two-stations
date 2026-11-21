"""Harness tests against an in-process mock of llama-server and vLLM (pass and gate-fail paths)."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from mock_server import Mock  # noqa: E402


def mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/f'{name}.py')
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


bench = mod('hy4bench')
an = mod('analyze')
FROZEN = ROOT/'data/frozen'


def frozen(name):
    return json.loads((FROZEN/name).read_text())


GSM = {tuple(x['no_think']): x['gold'] for x in frozen('gsm8k.json')}
GSM.update({tuple(x['high']): x['gold'] for x in frozen('gsm8k.json') if x['high']})
MMLU = {tuple(x['no_think']): x['gold'] for x in frozen('mmlu_pro.json')['items']}
NEEDLE = {tuple(x['ids']): x['codes'] for x in frozen('needle.json')}


def oracle(ids, wrong_every=0):
    t = tuple(ids)
    if t in GSM:
        return f'Some steps.\n#### {GSM[t]}'
    if t in MMLU:
        return f'Reasoning. The answer is ({MMLU[t]}).'
    if t in NEEDLE:
        return '\n'.join(f'code {i+1}: {c}' for i, c in enumerate(NEEDLE[t]))
    return 'lorem ipsum dolor sit amet'


_RUNS = iter(range(10**6))


def run(tmp_path, engine, cmd, *extra, **mock_kw):
    with Mock(engine, answer=mock_kw.pop('answer', oracle), **mock_kw) as m:
        out = tmp_path/f'{engine}-{cmd}-{next(_RUNS)}'
        rc = bench.main([cmd, '--base', m.base, '--engine', engine, '--rung', 'mock', '--out', str(out), *extra])
        return rc, out, m


def summary(out, name):
    return json.loads((out/f'{name}.summary.json').read_text())


# ------------------------------------------------------------------ frozen inputs

def test_frozen_manifest_hashes_match_files():
    import hashlib
    man = json.loads((FROZEN/'MANIFEST.json').read_text())
    for name, sha in man.items():
        if name.startswith('_'):
            continue
        assert hashlib.sha256((FROZEN/name).read_bytes()).hexdigest() == sha, name


def test_frozen_inputs_shape():
    t = frozen('templates.json')
    assert t['rendered']['no_think'].endswith('<think:opensource></think:opensource>')
    assert t['rendered']['high'].endswith('<think:opensource>')
    assert len(frozen('gsm8k.json')) == 500 and len(frozen('mmlu_pro.json')['items']) == 500
    assert sum(frozen('mmlu_pro.json')['quota'].values()) == 500
    for n in frozen('needle.json'):
        assert n['length']-300 < len(n['ids']) <= n['length']
    lp = frozen('logprob_corpus.json')
    assert {w['kind'] for w in lp} == {'text', 'code'}
    for w in lp:
        assert len(w['ids']) == 1024 and all(w['ids'][p] == t for p, t in zip(w['probes'], w['targets']))


# ------------------------------------------------------------------ scoring rules

@pytest.mark.parametrize('text,gold,ok', [('so #### 1,234', '1234', True), ('x = 5 then 7', '7', True),
                                          ('#### 7.', '7', True), ('#### 8', '7', False), ('no number', '3', False),
                                          ('#### -2', '-2', True)])
def test_gsm8k_scoring(text, gold, ok):
    assert bench.score_gsm8k(text, gold)[0] is ok


@pytest.mark.parametrize('text,gold,ok', [('The answer is (C).', 'C', True), ('answer is B', 'B', True),
                                          ('(A) no wait (D)', 'D', True), ('The answer is (C). Actually the answer is (E)', 'E', True),
                                          ('nothing', 'A', False)])
def test_mmlu_scoring(text, gold, ok):
    assert bench.score_mmlu(text, gold)[0] is ok


def test_needle_exact_match_is_token_bounded():
    assert bench.score_needle('code 1: ABCD1234\ncode 2: XYZ', ['ABCD1234', 'XYZ12345']) == [True, False]
    assert bench.score_needle('ABCD12345', ['ABCD1234']) == [False]


def test_thinking_on_scores_only_content():
    assert bench.answer_text('<think:opensource>#### 1</think:opensource>#### 2') == '#### 2'


def test_gguf_hashed_specials_are_handled_and_never_scored():
    # AngelSlim GGUFs render the same IDs as '<think:6124c78e>' / '<｜hy_end:6124c78e｜>' (seen 2026-11-20)
    assert bench.answer_text('<think:6124c78e>#### 1</think:6124c78e>#### 2<｜hy_end:6124c78e｜>') == '#### 2'
    assert bench.score_gsm8k(bench.strip_specials('The answer is 7.<｜hy_end:6124c78e｜>'), '7')[0]
    assert not bench.gate_thinking_off('x</think:6124c78e>y') and bench.gate_thinking_off('plain 6124 text')
    assert bench.has_think_close('a</think:opensource>b') and not bench.has_think_close('a</think>b')


# ------------------------------------------------------------------ pass paths, both engines

@pytest.mark.parametrize('engine', ['llamacpp', 'vllm'])
def test_speed_counts_tokens_from_usage_not_chunks(tmp_path, engine):
    rc, out, m = run(tmp_path, engine, 'speed', '--rounds', '2', '--warm', '1', '--decode-tokens', '32',
                     chunk_tokens=5)
    s = summary(out, 'speed')
    assert rc == 0 and s['c1']['all_gates_ok'] and s['c16']['all_gates_ok']
    rows = [json.loads(l) for l in (out/'speed.jsonl').read_text().splitlines()]
    assert all(r['completion_tokens'] == 32 for r in rows)  # usage, although only ceil(32/5) chunks arrived
    assert len(rows) == 3*16*2
    assert all(body.get('ignore_eos') and isinstance(body['prompt'], list) for p, body in m.requests if p == '/v1/completions')
    assert s['c16']['mean_tps'] > 0 and s['c1']['rounds'][1]['mean_decode_tps_per_reply'] > 0


def test_speed_heldout_prompts_separate_files(tmp_path):
    """Follow-up 1: --speed-prompts speed_heldout.json sends the held-out IDs and writes speed-heldout.*."""
    import json as _j
    from pathlib import Path as _P
    held = _j.loads((_P(__file__).resolve().parents[1]/'data/frozen/speed_heldout.json').read_text())
    rc, out, m = run(tmp_path, 'vllm', 'speed', '--rounds', '1', '--warm', '0', '--decode-tokens', '8',
                     '--speed-prompts', 'speed_heldout.json')
    assert rc == 0 and (out/'speed-heldout.jsonl').exists() and not (out/'speed.jsonl').exists()
    s = summary(out, 'speed-heldout'); assert s['c1']['all_gates_ok'] and s['c16']['all_gates_ok']
    sent = {tuple(b['prompt']) for p, b in m.requests if p == '/v1/completions'}
    assert sent == {tuple(x['ids']) for x in held} and len(held) == 16


@pytest.mark.parametrize('engine', ['llamacpp', 'vllm'])
def test_quality_needle_prefill_pass(tmp_path, engine):
    rc, out, _ = run(tmp_path, engine, 'quality', '--limit', '20', '--think')
    q = summary(out, 'quality')
    assert q['gsm8k:no_think']['accuracy'] == 1 and q['mmlu_pro:no_think']['accuracy'] == 1
    assert q['gsm8k:no_think']['gate_failures'] == 0
    rc, out, _ = run(tmp_path, engine, 'needle', '--max-needle', '32768')
    n = summary(out, 'needle')
    assert all(v['all_found'] for k, v in n.items() if '32k' in k) and all('skipped' in v for k, v in n.items() if '128k' in k)
    rc, out, m = run(tmp_path, engine, 'prefill', '--prefill-reps', '2')
    p = summary(out, 'prefill')
    assert all(v['gates_ok'] for v in p.values())
    sent = [b['prompt'] for path, b in m.requests if path == '/v1/completions']
    assert len({tuple(x[:40]) for x in sent}) == len(sent)  # every prefill request has a fresh nonce


def test_drafter_reports_acceptance_and_ms_per_step(tmp_path):
    rc, out, _ = run(tmp_path, 'vllm', 'drafter', '--max-draft', '16384', '--draft-reps', '2', '--draft-tokens', '64')
    s = summary(out, 'drafter')
    assert rc == 0 and set(s) == {'draft-2k', 'draft-16k'}
    for v in s.values():
        assert v['gates_ok'] and v['reps'] == 2 and v['ms_per_step_median'] > 0
        # 64 tokens: 63 after the first, 21 steps, 42 accepted of 105 drafted
        assert v['acceptance_median'] == pytest.approx(42/105) and v['mean_accepted_len_median'] == pytest.approx(3.0)


def test_drafter_frozen_lengths_fit_their_context():
    d = frozen('drafter.json')
    assert [x['length'] for x in d] == [2048, 16384, 131072, 524288]
    assert all(len(x['ids']) == x['prompt_tokens'] <= x['length']-256 for x in d)


def test_logprobs_survive_eog_greedy_token_on_llamacpp(tmp_path):
    # Real llama-server returns no completion_probabilities when the greedy token is end-of-generation.
    rc, out, _ = run(tmp_path, 'llamacpp', 'logprobs', '--limit', '1', '--topk', '16', eog_top1=True)
    _, ref, _ = run(tmp_path, 'llamacpp', 'logprobs', '--limit', '1', '--topk', '16')
    rows = [json.loads(l) for l in (out/'logprobs.jsonl').read_text().splitlines()]
    refrows = [json.loads(l) for l in (ref/'logprobs.jsonl').read_text().splitlines()]
    assert rc == 0 and rows and all(r['eog_retry'] and r['gate'] for r in rows)
    assert [r['top'] for r in rows] == [r['top'] for r in refrows]
    assert [r['sampled'] for r in rows] == [r['sampled'] for r in refrows]


def test_logprobs_record_incomplete_utf8_probes_and_continue(tmp_path):
    # llama-server never returns probs when the greedy token is a partial UTF-8 piece (probe 2761, 2026-11-20).
    rc, out, _ = run(tmp_path, 'llamacpp', 'logprobs', '--limit', '1', '--topk', '16', incomplete_utf8=True)
    rows = [json.loads(l) for l in (out/'logprobs.jsonl').read_text().splitlines()]
    s = summary(out, 'logprobs')
    assert rc == 0 and rows and all(not r['gate'] and r['no_probs'] == 'incomplete_utf8' and r['top'] == [] for r in rows)
    assert all(isinstance(r['sampled'], int) for r in rows)
    assert s['no_probs_incomplete_utf8'] == s['gate_failures'] == len(rows)


def test_logprob_capture_identical_positions_across_engines(tmp_path):
    _, ref, _ = run(tmp_path, 'vllm', 'logprobs', '--limit', '2', '--topk', '16')
    _, same, _ = run(tmp_path, 'llamacpp', 'logprobs', '--limit', '2', '--topk', '16')
    _, noisy, _ = run(tmp_path, 'llamacpp', 'logprobs', '--limit', '2', '--topk', '16', noise=1.0)
    assert summary(ref, 'logprobs')['gate_failures'] == 0
    d0 = an.drift(ref, same, 16)
    d1 = an.drift(ref, noisy, 16)
    assert d0['positions'] == 2*len(frozen('logprob_corpus.json')[0]['probes']) and d0['unmatched_positions'] == 0
    assert d0['all']['mean_kl'] == pytest.approx(0, abs=1e-9) and d0['all']['top1_agreement'] == 1
    assert d1['all']['mean_kl'] > 0.01 and d1['all']['top1_agreement'] < 1
    lo, hi = d1['all']['mean_kl_ci95']
    assert lo <= d1['all']['mean_kl'] <= hi


# ------------------------------------------------------------------ gate-fail paths

PROBE = frozen('templates.json')['alignment_probe']
KNOWN = [(PROBE['ids'], PROBE['text'])]+[(ids, s) for s, ids in frozen('templates.json')['specials'].items()]


@pytest.mark.parametrize('engine', ['llamacpp', 'vllm'])
def test_check_passes_with_matching_vocab(tmp_path, engine):
    rc, out, _ = run(tmp_path, engine, 'check', known_texts=KNOWN)
    s = summary(out, 'check')
    assert rc == 0 and s['prompt_gate'] and s['detok_match'] and s['tokenize_match']


HASHED = [(PROBE['ids'], PROBE['text'])]+[(ids, s.replace('opensource', '6124c78e'))
                                          for s, ids in frozen('templates.json')['specials'].items()]


def test_check_passes_with_gguf_hashed_special_names(tmp_path):
    rc, out, _ = run(tmp_path, 'llamacpp', 'check', known_texts=HASHED)
    s = summary(out, 'check')
    assert rc == 0 and s['detok_match'] and s['specials_match']
    assert s['specials_server']['</think:opensource>'] == '</think:6124c78e>'


def test_check_stops_when_a_special_id_is_not_a_single_token(tmp_path):
    bad = HASHED[:-1]  # last special's ID detokenizes to chr() junk that does not round-trip to its ID
    rc, out, _ = run(tmp_path, 'llamacpp', 'all', known_texts=bad)
    assert rc == 2 and not summary(out, 'check')['specials_match']


def test_vllm_requests_keep_special_tokens(tmp_path):
    _, _, m = run(tmp_path, 'vllm', 'check', known_texts=KNOWN)
    comp = [b for p, b in m.requests if p == '/v1/completions']
    assert comp and all(b['skip_special_tokens'] is False and b['spaces_between_special_tokens'] is False for b in comp)


def test_check_stops_on_vocab_mismatch(tmp_path):
    rc, out, _ = run(tmp_path, 'llamacpp', 'all')
    assert rc == 2 and not summary(out, 'check')['detok_match'] and not (out/'speed.summary.json').exists()


def test_check_stops_everything_when_prompt_tokens_misreported(tmp_path):
    rc, out, _ = run(tmp_path, 'llamacpp', 'all', misreport_prompt=True, known_texts=KNOWN)
    assert rc == 2 and not summary(out, 'check')['prompt_gate'] and not (out/'speed.summary.json').exists()


def test_missing_usage_fails_speed_gate_instead_of_counting_chunks(tmp_path):
    rc, out, _ = run(tmp_path, 'vllm', 'speed', '--rounds', '1', '--warm', '0', '--decode-tokens', '16', drop_usage=True)
    s = summary(out, 'speed')
    assert not s['c1']['all_gates_ok'] and not s['c16']['all_gates_ok']
    rows = [json.loads(l) for l in (out/'speed.jsonl').read_text().splitlines()]
    assert all(r['completion_tokens'] is None for r in rows)


def test_think_leak_and_short_reply_fail_gates(tmp_path):
    rc, out, _ = run(tmp_path, 'vllm', 'speed', '--rounds', '1', '--warm', '0', '--decode-tokens', '16', think_leak=True)
    assert not summary(out, 'speed')['c16']['all_gates_ok']
    rc, out, _ = run(tmp_path, 'llamacpp', 'speed', '--rounds', '1', '--warm', '0', '--decode-tokens', '16', short=True)
    assert not summary(out, 'speed')['c1']['all_gates_ok']
    rc, out, _ = run(tmp_path, 'llamacpp', 'quality', '--limit', '5', think_leak=True)
    assert summary(out, 'quality')['gsm8k:no_think']['gate_failures'] == 5


def test_recorder_refuses_to_append(tmp_path):
    rec = bench.Recorder(tmp_path, 'x'); rec.close()
    with pytest.raises(SystemExit):
        bench.Recorder(tmp_path, 'x')


def test_wrong_answers_are_scored_wrong(tmp_path):
    rc, out, _ = run(tmp_path, 'vllm', 'quality', '--limit', '10', answer=lambda ids: '#### 999999 The answer is (Z)')
    q = summary(out, 'quality')
    assert q['gsm8k:no_think']['accuracy'] == 0 and q['mmlu_pro:no_think']['accuracy'] == 0


# ------------------------------------------------------------------ statistics

def test_bucketed_kl_properties():
    p = {1: math.log(.5), 2: math.log(.3)}
    assert an.kl_bucketed(p, dict(p)) == pytest.approx(0, abs=1e-12)
    q = {1: math.log(.3), 2: math.log(.5)}
    assert an.kl_bucketed(p, q) > 0
    # a missing reference token gets the largest probability the test allows -> finite, positive
    assert 0 < an.kl_bucketed(p, {1: math.log(.6), 3: math.log(.2)}) < 10


def test_paired_quality_and_mcnemar(tmp_path):
    a = tmp_path/'a'; b = tmp_path/'b'; a.mkdir(); b.mkdir()
    rows_a = [dict(set='gsm8k', mode='no_think', id=f'q{i}', correct=i % 10 != 0) for i in range(100)]
    rows_b = [dict(set='gsm8k', mode='no_think', id=f'q{i}', correct=i % 5 != 0) for i in range(100)]
    (a/'quality.jsonl').write_text('\n'.join(json.dumps(r) for r in rows_a))
    (b/'quality.jsonl').write_text('\n'.join(json.dumps(r) for r in rows_b))
    r = an.paired(a, b, 'gsm8k:no_think')
    assert r['diff_b_minus_a'] == pytest.approx(-.10)
    lo, hi = r['newcombe95']
    assert lo < -.10 < hi < 0
    assert r['mcnemar_exact_p'] == pytest.approx(2/2**10)  # 10 discordant, all one way
    assert an.mcnemar(5, 5) == 1.0


def test_abba_ratio_interval():
    r = an.abba([[100, 105], [101, 106], [99, 104], [100, 105.5], [100, 104.5],
                 [102, 107], [98, 103], [100, 105], [101, 106], [99, 104]])
    assert r['df'] == 9 and r['t_crit'] == 2.262157
    assert 1.04 < r['ci95'][0] < r['ratio'] < r['ci95'][1] < 1.06
