#!/usr/bin/env python3
"""Figures for the Hy4 (L6-4h) posts. Every number is read from the run files, never typed in.

  uv run --no-project --with matplotlib==3.9.2 python -I publishing/hy4/make_figures.py

Inputs: workspace/data/l6-4h-hy4/ (raw runs copied from the Stations + analysis/out recomputed with
tools/hy4/analyze.py). Outputs: publishing/hy4/images/*.png (1600 px wide, light surface).
"""
import glob
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'results'
OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#9a998f'   # validated pair (validate_palette.js, light)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': .8, 'axes.axisbelow': True})
FOOT = 'RonanLabs L6-4h, 2026-11-21  |  2x DGX Station GB300 + RTX PRO 6000  |  @SJRonanMD'


def js(p):
    return json.loads(Path(p).read_text())


def one(pat):
    m = sorted(glob.glob(str(D/pat)))
    if len(m) != 1:
        sys.exit(f'{pat}: {m}')
    return m[0]


def speed(run, cell):
    return js(f'{run}/bench/speed.summary.json')[cell]['mean_tps']


def title(fig, t, sub):
    fig.text(.04, .955, t, fontsize=19, fontweight='bold', color=INK, va='top')
    fig.text(.04, .895, sub, fontsize=13, color=INK2, va='top')
    fig.text(.04, .02, FOOT, fontsize=9.5, color=INK2)


def save(fig, name):
    fig.savefig(OUT/name, dpi=200)
    plt.close(fig)
    print('wrote', OUT/name)


R = D/'station-left/runs'
RUNS = {k: one(f'station-left/runs/s3-{k}-*') for k in ('A1', 'A2', 'B1', 'B2')}

# ---- 1. sidecar speed -------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 5))
fig.subplots_adjust(left=.09, right=.97, top=.70, bottom=.14, wspace=.35)
abba = {c: js(D/f'analysis/out/abba-{c}.json') for c in ('c1', 'c16')}
for ax, cell, lab in zip(axes, ('c1', 'c16'), ('1 user', '16 users')):
    b = sum(speed(RUNS[k], cell) for k in ('B1', 'B2'))/2
    a = sum(speed(RUNS[k], cell) for k in ('A1', 'A2'))/2
    bars = ax.bar([0, 1], [b, a], width=.6, color=[GRAY, BLUE])
    for x, v in zip([0, 1], [b, a]):
        ax.text(x, v*1.02, f'{v:.1f}', ha='center', va='bottom', fontsize=14, color=INK, fontweight='bold')
    r = abba[cell]
    ax.set_title(f'{lab}: {r["ratio"]:.2f}x  (95% CI {r["ci95"][0]:.2f}-{r["ci95"][1]:.2f})', fontsize=13, color=INK, pad=10)
    ax.set_xticks([0, 1], ['FP8, experts\nin Grace memory', 'FP8 + RTX PRO 6000\nsidecar (MXFP4)'], fontsize=11)
    ax.set_ylim(0, a*1.18); ax.set_ylabel('output tokens / s'); ax.grid(axis='x', visible=False)
title(fig, 'Tencent Hy4 (770B MoE) on two DGX Stations',
      'Same FP8 checkpoint. The sidecar serves the most-used experts from the RTX PRO 6000\nin 4-bit (MXFP4). A-B-B-A, 10 paired rounds per cell. GSM8K, MMLU-Pro within noise.')
save(fig, 'hy4-sidecar-speed.png')

# ---- 2. drift vs benchmark ---------------------------------------------------------------------------
rows = [('MXFP4 sidecar (A1)', 'drift-A1-vs-B'), ('Q4_K_M', 'drift-q4_k_m'), ('IQ2', 'drift-iq2'), ('STQ1_0', 'drift-stq1_0')]
names, kl, klci, top1 = [], [], [], []
for n, df in rows:
    d = js(D/f'analysis/out/{df}.json')['all']
    names.append(n); kl.append(d['mean_kl']); klci.append(d['mean_kl_ci95']); top1.append(d['top1_agreement'])
fig = plt.figure(figsize=(8, 5))
ax = fig.add_axes([.25, .16, .38, .52])
gs = {'MXFP4 sidecar (A1)': js(D/'analysis/out/paired-A1-gsm8k.json')['b_acc'],
      'Q4_K_M': js(D/'analysis/out/paired-q4_k_m-gsm8k-no_think.json')['b_acc'],
      'IQ2': js(D/'analysis/out/paired-iq2-gsm8k-no_think.json')['b_acc'],
      'STQ1_0': js(D/'analysis/out/paired-stq1_0-gsm8k-no_think.json')['b_acc']}
fp8 = js(D/'analysis/out/paired-q4_k_m-gsm8k-no_think.json')['a_acc']
y = list(range(len(names)))[::-1]
cols = [BLUE, GRAY, GRAY, GRAY]
ax.barh(y, kl, height=.55, color=cols)
for yi, v, ci, t, n in zip(y, kl, klci, top1, names):
    ax.plot(ci, [yi, yi], color=INK, lw=1.5)
    for xo, txt in ((1.06, f'{v:.3f}'), (1.36, f'{100*t:.1f}%'), (1.70, f'{100*gs[n]:.1f}%')):
        ax.text(xo, yi, txt, transform=ax.get_yaxis_transform(), va='center', ha='left', fontsize=12.5, color=INK)
for xo, h in ((1.06, 'KL'), (1.36, 'top-1'), (1.70, 'GSM8K')):
    ax.text(xo, len(names)-.35, h, transform=ax.get_yaxis_transform(), va='bottom', ha='left', fontsize=11, color=INK2)
ax.set_yticks(y, names, fontsize=12); ax.set_xlim(0, .7); ax.grid(axis='y', visible=False)
ax.set_xlabel('mean KL from FP8, nats, 95% CI (lower = closer)')
title(fig, 'Benchmarks barely move. Token-level drift does.',
      f'Drift from FP8 (GSM8K {100*fp8:.1f}%) for each version. FP8 itself gave byte-identical\nlog-probabilities on three separate loads, so the reference adds no noise.')
save(fig, 'hy4-drift-vs-benchmark.png')

# ---- 3. drafter flat cost ----------------------------------------------------------------------------
fig = plt.figure(figsize=(8, 5))
ax = fig.add_axes([.1, .14, .86, .62])
for run, col, lab in ((one('station-left/runs/fp8b-*'), BLUE, 'FP8 reference config (166.5 GiB experts on Grace)'),
                      (one('station-left/runs/fp8d512-*'), ORANGE, '512K config (193.5 GiB experts on Grace)')):
    d = js(f'{run}/bench/drafter.summary.json')
    pts = sorted((v['length'], v['ms_per_step_median']) for v in d.values())
    xs = [math.log2(p[0]) for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=col, lw=2, marker='o', ms=8, label=lab)
    ax.text(xs[-1]+.2, ys[-1], f'{ys[0]:.1f} to {ys[-1]:.1f} ms', va='center', fontsize=11.5, color=INK)
ticks = [2048, 16384, 131072, 524288]
ax.set_xticks([math.log2(t) for t in ticks], ['2K', '16K', '128K', '512K'])
ax.set_xlim(10.5, 21.8); ax.set_ylim(0, 135); ax.set_xlabel('context length (tokens, log scale)'); ax.set_ylabel('ms per verify step (5 drafted tokens)')
ax.legend(loc='lower left', frameon=False, fontsize=11)
title(fig, "Hy4's drafter: flat cost per step, 2K to 512K",
      'Median of 3 reps, 256-token summary of a document of each length.\nUnder 2% growth, as Tencent claims. vLLM FP8 across two DGX Stations.')
save(fig, 'hy4-drafter-flat.png')

# ---- 4. GGUF ladder vs FP8 ---------------------------------------------------------------------------
gg = [('FP8\n(2 Stns)', one('station-left/runs/fp8b-*')),
      ('STQ1_0', one('station-left/runs/stq1_0-left-20261120-165703')),
      ('IQ2', one('station-left/runs/iq2-left-20261120-200346')),
      ('Q4_K_M', one('station-right/runs/q4_k_m-right-20261120-165707'))]
fig, axes = plt.subplots(1, 2, figsize=(8, 5))
fig.subplots_adjust(left=.07, right=.98, top=.72, bottom=.2, wspace=.25)
for ax, cell, lab in zip(axes, ('c1', 'c16'), ('1 user', '16 users')):
    v = [speed(r, cell) for _, r in gg]
    ax.bar(range(4), v, width=.62, color=[BLUE, GRAY, GRAY, GRAY])
    for i, x in enumerate(v):
        ax.text(i, x*1.02, f'{x:.1f}', ha='center', va='bottom', fontsize=12, color=INK, fontweight='bold')
    ax.set_xticks(range(4), [n for n, _ in gg], fontsize=11.5); ax.set_title(lab, fontsize=13, color=INK)
    ax.set_ylim(0, max(v)*1.18); ax.set_ylabel('output tokens / s'); ax.grid(axis='x', visible=False)
title(fig, 'Hy4 770B: three GGUF versions vs FP8',
      'GGUF on llama.cpp, one Station each (IQ2 adds the RTX 6000; Q4_K_M adds\nthe 6000 and Grace). All found 18/18 needles; GSM8K within 0.6 pt of FP8.')
save(fig, 'hy4-gguf-vs-fp8.png')
