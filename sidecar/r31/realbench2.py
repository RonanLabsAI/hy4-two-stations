"""realbench2.py <base> <model> <out.json> <C> <n> [seed_offset] -- R2.3 realbench.py with TOKEN-accurate TPOT.
Same 36 held-out ACS h2 prompts, thinking off, temperature 0, max_tokens 1024 + ignore_eos. With speculative decoding a
streamed chunk can carry several tokens, so TPOT = (t_last - t_first) / (completion_tokens - 1) using the server's usage
count (stream_options.include_usage), not the chunk count. Also records chunks so tokens/chunk is visible.
seed_offset rotates which prompt each request gets (prompt i = (i + offset) % 36) so reps do not replay the same order.
"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE, MODEL, OUT = sys.argv[1:4]
C, N = int(sys.argv[4]), int(sys.argv[5]); OFF = int(sys.argv[6]) if len(sys.argv) > 6 else 0
TOPICS = ["hypoxia and its four types", "hyperventilation in flight", "spatial disorientation and vestibular illusions",
          "the effects of alcohol and drugs on pilot performance", "carbon monoxide poisoning from cabin heaters",
          "decompression sickness after scuba diving", "aeronautical decision-making and hazardous attitudes",
          "fatigue and circadian rhythm in flight operations", "middle ear and sinus blockage during descent",
          "visual scanning and collision avoidance", "dehydration and heatstroke in the cockpit",
          "single-pilot resource management and task saturation"]
FORMS = ["Explain {t} to a private pilot student in detail, with practical examples and how the FAA ACS tests it.",
         "Write a thorough study guide section on {t} for a CFI to use in ground school.",
         "Describe the causes, symptoms, prevention and in-flight response for {t}, citing typical FAA guidance."]
PROMPTS = [f.format(t=t) for t in TOPICS for f in FORMS]


def one(i):
    body = {'model': MODEL, 'temperature': 0.0, 'max_tokens': 1024, 'ignore_eos': True, 'stream': True,
            'stream_options': {'include_usage': True}, 'chat_template_kwargs': {'thinking': False},
            'messages': [{'role': 'user', 'content': PROMPTS[(i + OFF) % len(PROMPTS)]}]}
    req = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time(); first = last = None; n = 0; toks = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            d = json.loads(line[5:])
            if d.get('usage'):
                toks = d['usage'].get('completion_tokens')
            ch = (d.get('choices') or [{}])[0]
            delta = ch.get('delta', {})
            if delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning'):
                now = time.time(); first = first or now; last = now; n += 1
    toks = toks or n
    return {'ttft': first - t0, 'chunks': n, 'tokens': toks, 'tpot_ms': 1000 * (last - first) / max(toks - 1, 1),
            'wall': last - t0}


t0 = time.time()
with ThreadPoolExecutor(C) as ex:
    res = list(ex.map(one, range(N)))
wall = time.time() - t0
mt = sum(r['tpot_ms'] for r in res) / len(res)
summ = {'C': C, 'n': N, 'offset': OFF, 'mean_tpot_ms': round(mt, 2), 'decode_agg': round(C * 1000 / mt, 2),
        'median_ttft_s': round(sorted(r['ttft'] for r in res)[len(res) // 2], 2),
        'mean_tokens': sum(r['tokens'] for r in res) / N, 'tok_per_chunk': round(sum(r['tokens'] for r in res) / max(1, sum(r['chunks'] for r in res)), 3),
        'wall_s': round(wall, 1)}
json.dump({'summary': summ, 'req': res}, open(OUT, 'w'))
print(json.dumps(summ))
