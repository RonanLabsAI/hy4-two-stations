"""In-process mock of llama-server and vLLM OpenAI-compatible endpoints for harness tests.

Fake model: next-token logprobs are a deterministic function of (last 3 prompt ids, noise seed). `noise`
perturbs them (a stand-in for a lower-precision rung). Replies are chosen by an `answer` callback given
the prompt ids, so tests can make quality/needle items right or wrong. Knobs simulate failure modes:
misreport_prompt (gate G1), drop_usage, think_leak (gate G2), short (gate G3).
"""
from __future__ import annotations

import json
import math
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VOCAB = 2000


def logprobs_for(ids, noise=0.0, seed=0):
    rng = random.Random(hash((tuple(ids[-3:]), -1)) & 0xffffffff)
    logits = [rng.gauss(0, 3) for _ in range(VOCAB)]
    if noise:
        nrng = random.Random(hash((tuple(ids[-3:]), seed)) & 0xffffffff)
        logits = [x+nrng.gauss(0, noise) for x in logits]
    m = max(logits)
    z = m+math.log(sum(math.exp(x-m) for x in logits))
    return [x-z for x in logits]


class Mock:
    def __init__(self, engine, answer=None, noise=0.0, misreport_prompt=False, drop_usage=False,
                 think_leak=False, short=False, chunk_tokens=3, delay_per_token=0.0005, known_texts=None,
                 eog_top1=False, incomplete_utf8=False):
        self.eog_top1 = eog_top1; self.incomplete_utf8 = incomplete_utf8
        self.spec = {'drafts': 0, 'draft_tokens': 0, 'accepted_tokens': 0}  # fake DFly2: 3 tokens per verify step
        self.engine = engine; self.answer = answer or (lambda ids: 'ok')
        self.noise = noise; self.misreport_prompt = misreport_prompt; self.drop_usage = drop_usage
        self.think_leak = think_leak; self.short = short; self.chunk_tokens = chunk_tokens
        self.delay = delay_per_token; self.requests = []; self.cache = []
        self.known = {tuple(k): v for k, v in (known_texts or [])}  # ids -> text, to mimic the real vocab
        self.lock = threading.Lock()
        mock = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj, code=200):
                data = json.dumps(obj).encode()
                self.send_response(code); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

            def do_GET(self):
                if self.path == '/v1/models':
                    return self._json({'data': [{'id': f'mock-{mock.engine}'}]})
                if self.path == '/metrics':
                    sp = mock.spec
                    data = ('# mock metrics\n' + ''.join(f'vllm:spec_decode_num_{k}_total{{model_name="m",engine="0"}} {v}\n'
                                                          for k, v in sp.items())).encode()
                    self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers()
                    return self.wfile.write(data)
                self._json({'error': 'not found'}, 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with mock.lock:
                    mock.requests.append((self.path, body))
                if self.path == '/tokenize':
                    text = body.get('content', body.get('prompt', ''))
                    for ids, t in mock.known.items():
                        if t == text:
                            return self._json({'tokens': list(ids)})
                    return self._json({'tokens': [ord(ch) % VOCAB for ch in text]})
                if self.path == '/detokenize':
                    toks = body['tokens']
                    key = 'content' if mock.engine == 'llamacpp' else 'prompt'
                    return self._json({key: mock.known.get(tuple(toks), ''.join(chr(t) for t in toks))})
                if self.path == '/v1/completions':
                    return self.completions(body)
                if self.path == '/completion' and mock.engine == 'llamacpp':
                    return self.native(body)
                self._json({'error': 'not found'}, 404)

            def reply_tokens(self, ids, max_tokens, ignore_eos):
                text = mock.answer(ids)
                if mock.think_leak:
                    text = '</think:opensource>'+text
                n = max_tokens if ignore_eos else min(max_tokens, max(1, len(text.split())))
                if mock.short and ignore_eos:
                    n = max_tokens-1
                return text, n

            def usage(self, ids, n):
                if mock.drop_usage:
                    return None
                cached = 0
                with mock.lock:
                    for prev in mock.cache:
                        k = 0
                        while k < min(len(prev), len(ids)) and prev[k] == ids[k]:
                            k += 1
                        cached = max(cached, k)
                    mock.cache.append(list(ids))
                return {'prompt_tokens': len(ids)+(1 if mock.misreport_prompt else 0), 'completion_tokens': n,
                        'total_tokens': len(ids)+n, 'prompt_tokens_details': {'cached_tokens': cached}}

            def completions(self, body):
                ids = body['prompt']; assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
                text, n = self.reply_tokens(ids, body['max_tokens'], body.get('ignore_eos', False))
                if mock.engine == 'vllm':
                    steps = math.ceil(max(n-1, 0)/3)
                    with mock.lock:
                        mock.spec['drafts'] += steps; mock.spec['draft_tokens'] += 5*steps
                        mock.spec['accepted_tokens'] += max(n-1, 0)-steps
                import time
                if body.get('stream'):
                    self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
                    pieces = max(1, math.ceil(n/mock.chunk_tokens))
                    for i in range(pieces):
                        time.sleep(mock.delay*mock.chunk_tokens)
                        piece = text if i == 0 else ' .'
                        ev = {'choices': [{'index': 0, 'text': piece, 'finish_reason': 'length' if i == pieces-1 else None}]}
                        self.wfile.write(f'data: {json.dumps(ev)}\n\n'.encode())
                    u = self.usage(ids, n)
                    if u and body.get('stream_options', {}).get('include_usage'):
                        self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': u})}\n\n".encode())
                    self.wfile.write(b'data: [DONE]\n\n')
                    return
                time.sleep(mock.delay*n)
                ch = {'index': 0, 'text': text, 'finish_reason': 'length'}
                if body.get('logprobs'):
                    k = body['logprobs']
                    lp = logprobs_for(ids, mock.noise)
                    top = sorted(range(VOCAB), key=lambda i: -lp[i])[:k]
                    fmt = (lambda i: f'token_id:{i}') if body.get('return_tokens_as_token_ids') else (lambda i: chr(i))
                    ch['logprobs'] = {'tokens': [fmt(top[0])], 'token_logprobs': [lp[top[0]]],
                                      'top_logprobs': [{fmt(i): lp[i] for i in top}], 'text_offset': [0]}
                self._json({'choices': [ch], 'usage': self.usage(ids, n)})

            def native(self, body):
                ids = body['prompt']; k = body.get('n_probs', 0)
                lp = logprobs_for(ids, mock.noise)
                top = sorted(range(VOCAB), key=lambda i: -lp[i])[:k]
                u = self.usage(ids, 1)
                if mock.incomplete_utf8:  # greedy token is a partial UTF-8 piece: no probs even with ignore_eos
                    return self._json({'content': '', 'tokens': [top[0]], 'tokens_predicted': 1,
                                       'tokens_evaluated': u['prompt_tokens'] if u else None})
                if mock.eog_top1 and not body.get('ignore_eos'):  # greedy token is EOG: real server sends no probs
                    return self._json({'content': '', 'tokens_evaluated': u['prompt_tokens'] if u else None,
                                       'tokens_predicted': 0})
                self._json({'content': chr(top[0]), 'tokens_evaluated': u['prompt_tokens'] if u else None,
                            'tokens_predicted': 1,
                            'completion_probabilities': [{'id': top[0], 'token': chr(top[0]), 'logprob': lp[top[0]],
                                                          'top_logprobs': [{'id': i, 'token': chr(i), 'logprob': lp[i]} for i in top]}]})

        ThreadingHTTPServer.request_queue_size = 128
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.base = f'http://127.0.0.1:{self.server.server_address[1]}'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self, *a):
        self.server.shutdown(); self.server.server_close()
