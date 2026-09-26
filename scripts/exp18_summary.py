#!/usr/bin/env python3
"""
exp18_summary.py - collect every pre-clamp multi-step result (exp17 --preclamp) into the
synthetic 3x3 and real 2x2 transfer tables, medians [IQR] over seeds, and draw
figL (synthetic) and figM (real). Writes results/result_exp18_summary.json and
reports/exp18_summary.md. Synthetic arrival is converted to minutes (1 frame = 2 min).
Safe to rerun: it rebuilds the summary from whatever results exist.
"""
import os, re, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, FIG = os.path.join(ROOT, 'results'), os.path.join(ROOT, 'results', 'figures')
SYN, REAL = ['point', 'distributed', 'inflow'], ['breach', 'harvey']
COL = {'point': '#7b3294', 'distributed': '#1b7837', 'inflow': '#e08214',
       'breach': '#1F6FB2', 'harvey': '#C9522A'}
INK, MUTED, GOOD, PAPER = '#1A2332', '#6B7A8F', '#2E7D5B', '#FBFAF7'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'axes.grid': True, 'grid.color': '#D8DEE6',
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'legend.frameon': False,
                     'figure.facecolor': PAPER, 'axes.facecolor': PAPER, 'savefig.facecolor': PAPER})


def load_runs():
    runs = {}
    for f in glob.glob(os.path.join(R, 'result_exp17_ms_*_k8_pc_s*.json')):
        m = re.search(r'result_exp17_ms_(point|distributed|inflow|breach|harvey)_k8_pc_s(\d+)\.json', f)
        if m:
            runs.setdefault(m.group(1), {})[int(m.group(2))] = json.load(open(f))
    return runs


def med_iqr(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return None
    return [float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))]


def fmt(m, d=3, suffix=''):
    return '-' if m is None else f"{m[0]:.{d}f}{suffix} [{m[1]:.{d}f}, {m[2]:.{d}f}]"


def table(runs, trainers, tests, synth):
    out, lines = {}, []
    unit = 'min' if synth else 'h'
    lines.append(f"| Trained on | Tested on | Seeds | CSI | All-wet CSI | S | ETS | Volume ratio | Arrival ({unit}) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for tr in trainers:
        rs = runs.get(tr, {})
        for te in tests:
            sc = [rs[s]['scores'][te] for s in sorted(rs) if te in rs[s]['scores']]
            cell = {k: med_iqr([x[k] for x in sc]) for k in ('csi', 'allwet_csi', 'S', 'ets', 'volume_ratio')}
            cell['arrival'] = med_iqr([x['arrival_h'] * (2.0 if synth else 1.0) for x in sc])
            cell['n'] = len(sc)
            cell['gate_pass'] = sum(bool(rs[s].get('gate')) for s in rs) if te == tr else None
            out[f'{tr}->{te}'] = cell
            name = f"{te} (own)" if te == tr else te
            lines.append(f"| {tr} | {name} | {len(sc)} | {fmt(cell['csi'])} | {fmt(cell['allwet_csi'])} | "
                         f"{fmt(cell['S'])} | {fmt(cell['ets'])} | {fmt(cell['volume_ratio'], 2, 'x')} | "
                         f"{fmt(cell['arrival'], 1)} |")
    return out, lines


def plot(runs, trainers, tests, name, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pos = {(tr, te): gi * (len(tests) + 1) + ti for gi, tr in enumerate(trainers) for ti, te in enumerate(tests)}
    specs = [('csi', 'CSI (1% relative threshold)', False), ('ets', 'ETS (0 = chance)', False),
             ('volume_ratio', 'Volume ratio (predicted / true)', True)]
    for ax, (k, lab, log) in zip(axes, specs):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for (tr, te), x in pos.items():
            rs = runs.get(tr, {})
            v = [rs[s]['scores'][te][k] for s in rs if te in rs[s]['scores']]
            v = [x_ for x_ in v if x_ is not None and np.isfinite(x_)]
            if not v:
                continue
            ax.scatter([x] * len(v), v, s=16, color=COL[te], alpha=0.35, zorder=2)
            ax.scatter([x], [np.median(v)], s=80, color=COL[te], zorder=3,
                       edgecolor=INK if tr == te else 'none', linewidth=1.8)
            if k == 'csi':
                aw = [rs[s]['scores'][te]['allwet_csi'] for s in rs if te in rs[s]['scores']]
                ax.hlines(np.median(aw), x - 0.4, x + 0.4, colors=INK, linestyles=':', lw=1.4, zorder=1)
        if log:
            ax.set_yscale('log')
            ax.axhspan(0.5, 2.0, color=GOOD, alpha=0.10, zorder=0)
            ax.axhline(1.0, color=GOOD, lw=1, ls='--')
        if k == 'ets':
            ax.axhline(0, color=MUTED, lw=1.2)
        ax.set_ylabel(lab)
        ax.set_xticks(list(pos.values()))
        ax.set_xticklabels([te for (tr, te) in pos], rotation=35, fontsize=8)
        for gi, tr in enumerate(trainers):
            ax.text(gi * (len(tests) + 1) + (len(tests) - 1) / 2, -0.30, f'trained on {tr}',
                    transform=ax.get_xaxis_transform(), ha='center', fontsize=9, color=INK)
    handles = ([Line2D([], [], marker='o', ls='', color=COL[t], label=f'tested on {t}') for t in tests]
               + [Line2D([], [], marker='o', ls='', color='white', markeredgecolor=INK, label='own flood type'),
                  Line2D([], [], ls=':', color=INK, label='CSI of flooding every cell'),
                  Patch(color=GOOD, alpha=0.10, label='volume within 2x')])
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=8.5, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(title, fontsize=13.5, fontweight='bold', x=0.05, ha='left')
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {name}.png/.pdf")


def main():
    runs = load_runs()
    print("runs found:", {d: sorted(s) for d, s in runs.items()})
    summary, md = {}, ["# exp18 - trained models (multi-step, source channel, pre-clamp loss)", ""]
    for label, trainers, tests, synth in (('Synthetic 3 x 3 (30-step rollout, 15 events; arrival in minutes)', SYN, SYN, True),
                                          ('Real 2 x 2 (40-step rollout, 19 events; arrival in hours)', REAL, REAL, False)):
        out, lines = table(runs, trainers, tests, synth)
        summary.update(out)
        md += [f"## {label}", ""] + lines + [""]
        print(f"\n{label}")
        print("\n".join(lines))
        gates = {tr: f"{sum(bool(runs[tr][s].get('gate')) for s in runs[tr])}/{len(runs[tr])}"
                 for tr in trainers if tr in runs}
        print("  seeds passing the gate:", gates)
        md += [f"Seeds passing the home gate: {gates}", ""]
    json.dump(summary, open(os.path.join(R, 'result_exp18_summary.json'), 'w'), indent=2)
    open(os.path.join(ROOT, 'reports', 'exp18_summary.md'), 'w').write("\n".join(md))
    print("\n  saved results/result_exp18_summary.json and reports/exp18_summary.md")
    plot(runs, SYN, SYN, 'figL_trained_synthetic', 'Trained models: three controlled flood mechanisms')
    if any(d in runs for d in REAL):
        plot(runs, REAL, REAL, 'figM_trained_real', 'Trained models: dam-break and rainfall floods')


if __name__ == '__main__':
    main()
