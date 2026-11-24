#!/usr/bin/env python3
"""L6-4g Part 3 (Amendment 2): fetch AtCoder task statements into tools/hy4/data/part3/tasks/<task_id>/.

Per task writes:
  statement.txt  English statement as plain text (math kept as written inside $...$, samples included)
  meta.json      contest, task id, title, time limit (s), memory limit (MiB), interactive flag, source URL, sha256
  samples/NN.in, samples/NN.out

Statements are not republished (Amendment 2); only IDs and hashes go public.
Usage: fetch_tasks.py abc483 agc079 abc484 abc485 arc234
"""
import hashlib, html, json, re, sys, time, urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / 'data' / 'part3' / 'tasks'
UA = {'User-Agent': 'Mozilla/5.0 (research benchmark fetcher)'}


def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read().decode('utf-8')
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))


class EnText(HTMLParser):
    """Collect plain text of <span class="lang-en"> with light markup."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0  # span depth inside lang-en; 0 = outside
        self.out = []
        self.pre = 0
        self.var = False
        self.li = 0
        self.cur_h3 = None
        self.sections = []  # (title, text) for samples

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if self.depth == 0:
            if tag == 'span' and 'lang-en' in (a.get('class') or '').split():
                self.depth = 1
            return
        if tag == 'span':
            self.depth += 1
        elif tag == 'h3':
            self.out.append('\n\n## ')
            self.cur_h3 = []
        elif tag == 'pre':
            self.pre += 1
            self.out.append('\n```\n')
            self.pre_buf_start = len(self.out)
        elif tag == 'var':
            self.var = True
            if not self.pre:
                self.out.append('$')
        elif tag == 'li':
            self.out.append('\n- ')
        elif tag in ('p', 'div', 'ul', 'ol'):
            self.out.append('\n\n' if tag == 'p' else '\n')
        elif tag == 'br':
            self.out.append('\n')
        elif tag == 'tr':
            self.out.append('\n|')
        elif tag in ('td', 'th'):
            self.out.append(' ')
        elif tag == 'img':
            self.out.append('[image: %s]' % a.get('src', ''))

    def handle_endtag(self, tag):
        if self.depth == 0:
            return
        if tag == 'span':
            self.depth -= 1
        elif tag == 'h3':
            self.cur_h3 = ''.join(self.cur_h3).strip()
        elif tag == 'pre':
            body = ''.join(self.out[self.pre_buf_start:])
            self.sections.append((self.cur_h3, body))
            self.pre -= 1
            if not body.endswith('\n'):
                self.out.append('\n')
            self.out.append('```\n')
        elif tag == 'var':
            self.var = False
            if not self.pre:
                self.out.append('$')
        elif tag in ('td', 'th'):
            self.out.append(' |')
        elif tag == 'table':
            self.out.append('\n')

    def handle_data(self, data):
        if self.depth == 0:
            return
        if isinstance(self.cur_h3, list):
            self.cur_h3.append(data)
        if not self.pre:
            data = re.sub(r'[ \t\r\n]+', ' ', data)
        self.out.append(data)

    def text(self):
        t = ''.join(self.out).replace('\r\n', '\n').replace('\r', '\n')  # samples may carry CRLF
        t = re.sub(r'[ \t]+\n', '\n', t)
        t = re.sub(r'\n{3,}', '\n\n', t)
        return t.strip() + '\n'


def task_ids(contest):
    page = get(f'https://atcoder.jp/contests/{contest}/tasks?lang=en')
    ids = []
    for m in re.finditer(rf'href="/contests/{contest}/tasks/({contest}_[a-z0-9]+)"', page):
        if m.group(1) not in ids:
            ids.append(m.group(1))
    return ids


def fetch(contest, tid):
    url = f'https://atcoder.jp/contests/{contest}/tasks/{tid}?lang=en'
    page = get(url)
    m = re.search(r'Time Limit:\s*([\d.]+)\s*sec\s*/\s*Memory Limit:\s*(\d+)\s*MiB', page)
    tl, ml = float(m.group(1)), int(m.group(2))
    title = html.unescape(re.search(r'<span class="h2">\s*(.*?)\s*(?:<a|</span>)', page, re.S).group(1)).strip()
    p = EnText()
    p.feed(page)
    text = p.text()
    interactive = bool(re.search(r'interactive|Interaction', text, re.I))
    d = ROOT / tid
    (d / 'samples').mkdir(parents=True, exist_ok=True)
    ins = [b for h, b in p.sections if h and h.startswith('Sample Input')]
    outs = [b for h, b in p.sections if h and h.startswith('Sample Output')]
    for i, (a, b) in enumerate(zip(ins, outs), 1):
        (d / 'samples' / f'{i:02d}.in').write_text(a.lstrip('\n'))
        (d / 'samples' / f'{i:02d}.out').write_text(b.lstrip('\n'))
    (d / 'statement.txt').write_text(text)
    meta = dict(contest=contest, task_id=tid, title=title, time_limit_s=tl, memory_limit_mib=ml,
                interactive=interactive, n_samples=min(len(ins), len(outs)), url=url,
                statement_sha256=hashlib.sha256(text.encode()).hexdigest())
    (d / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    return meta


def main():
    for contest in sys.argv[1:]:
        for tid in task_ids(contest):
            m = fetch(contest, tid)
            print(f"{tid}\t{m['time_limit_s']}s\t{m['memory_limit_mib']}MiB\tsamples={m['n_samples']}"
                  f"\tinteractive={m['interactive']}\t{m['title']}")
            time.sleep(1.5)


if __name__ == '__main__':
    main()
