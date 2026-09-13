"""
fix_figA_disagreement.py
=========================

Replaces the Spearman-rho box on figA with the pairwise-disagreement
percentages, which are the number that survived review. The rho values
were dropped from the paper because, read plainly, a negative rho in
the paradox direction means "higher CSI correlates with better volume
accuracy" -- the opposite of the intended point -- while the raw
disagreement count and the aggregate median contrast make the point
correctly and are not sensitive to this issue.

Run:
    python3 scripts/fix_figA_disagreement.py
"""

import itertools
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = 'results'
FIGDIR = 'results/figures'

INK = '#1A2332'
MUTED = '#6B7A8F'
BREACH = '#1F6FB2'
HARVEY = '#C9522A'
GOOD = '#2E7D5B'
DANGER = '#A62B2B'
PAPER = '#FBFAF7'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'axes.titlecolor': INK,
    'axes.linewidth': 0.9, 'axes.grid': True, 'grid.color': '#D8DEE6',
    'grid.linewidth': 0.6, 'grid.alpha': 0.8,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'legend.frameon': False,
    'figure.facecolor': PAPER, 'axes.facecolor': PAPER,
    'savefig.facecolor': PAPER,
})


def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ('png', 'pdf'):
        p = os.path.join(FIGDIR, f'{name}.{ext}')
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f"  saved {p}")
    plt.close(fig)


def pairwise_disagreement(events):
    """% of event pairs where CSI ranking and volume-accuracy ranking disagree."""
    total, dis = 0, 0
    for a, b in itertools.combinations(events, 2):
        ca, cb = a['csi_final'], b['csi_final']
        if ca == cb:
            continue
        va = abs(math.log10(max(a['volume_ratio'], 1e-6)))
        vb = abs(math.log10(max(b['volume_ratio'], 1e-6)))
        if va == vb:
            continue
        csi_a_better = ca > cb
        vol_a_better = va < vb
        total += 1
        if csi_a_better != vol_a_better:
            dis += 1
    return dis, total, (100.0 * dis / total if total else float('nan'))


def main():
    with open(os.path.join(RESULTS, 'result_exp6_metric_disagreement.json')) as f:
        data = json.load(f)

    fig, (ax, axh) = plt.subplots(
        1, 2, figsize=(13.5, 7.6),
        gridspec_kw={'width_ratios': [3.6, 1], 'wspace': 0.05})
    style_axes(ax)
    style_axes(axh)

    series = [
        ('breach_home', BREACH, 'o', True, 'Dam-break model, own data'),
        ('breach_away', BREACH, 'o', False, 'Dam-break model, rainfall data'),
        ('harvey_home', HARVEY, 's', True, 'Rainfall model, own data'),
        ('harvey_away', HARVEY, 's', False, 'Rainfall model, dam-break data'),
    ]

    ax.axhspan(0.5, 2.0, color=GOOD, alpha=0.10, zorder=0)
    ax.axhline(1.0, color=GOOD, lw=1.4, ls='--', zorder=1)

    for key, colour, marker, filled, label in series:
        if key not in data:
            continue
        ev = data[key]['events']
        ax.scatter([e['csi_final'] for e in ev],
                   [max(e['volume_ratio'], 1e-4) for e in ev],
                   marker=marker, s=95,
                   facecolors=colour if filled else 'none',
                   edgecolors=colour, linewidths=1.8,
                   alpha=0.85 if filled else 1.0, zorder=4, label=label)

    ax.set_yscale('log')
    ax.set_xlim(0, 1.02)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 16)

    ax.text(0.015, 1.13, 'perfect water balance', color=GOOD, fontsize=9.5,
            style='italic', va='bottom')

    if 'breach_away' in data:
        worst = max(data['breach_away']['events'],
                    key=lambda e: e['volume_ratio'])
        ax.annotate(
            f"event {worst['event_index']}:  CSI {worst['csi_final']:.2f} "
            f"looks acceptable,\nyet it predicts "
            f"{worst['volume_ratio']:.0f} times the true water",
            xy=(worst['csi_final'], worst['volume_ratio']),
            xytext=(0.03, hi * 5.5),
            fontsize=10.5, color=DANGER, fontweight='bold', ha='left',
            arrowprops=dict(arrowstyle='->', color=DANGER, lw=1.7,
                            connectionstyle='arc3,rad=-0.18'), zorder=7)

    # ---- the fixed box: disagreement %, median CSI/volume, bad-event count ----
    rows = []
    for key, _, _, _, label in series:
        if key not in data:
            continue
        ev = data[key]['events']
        csis = [e['csi_final'] for e in ev]
        vrs = [e['volume_ratio'] for e in ev]
        d, t, pct = pairwise_disagreement(ev)
        bad = data[key]['n_bad_disagreements']
        n = data[key]['n_events']
        rows.append(
            f"{label}:\n"
            f"  median CSI {np.median(csis):.3f}, median volume {np.median(vrs):.2f}x  |  "
            f"{pct:.1f}% of event pairs disagree  |  {bad}/{n} good-CSI/bad-volume"
        )
    # stats box moved below the axes entirely - see fig.text calls at the bottom

    ax.set_xlabel('CSI - the metric the field reports (higher looks better)',
                  fontsize=11.5)
    ax.set_ylabel('Volume ratio, log scale\n(predicted water / true water)',
                  fontsize=11.5)
    ax.set_title('Good CSI does not mean the water is right',
                 fontsize=15, fontweight='bold', loc='left', pad=16)
    ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.11),
              ncol=4, columnspacing=1.4, handletextpad=0.4)

    axh.set_yscale('log')
    axh.axhspan(0.5, 2.0, color=GOOD, alpha=0.10, zorder=0)
    axh.axhline(1.0, color=GOOD, lw=1.4, ls='--', zorder=1)
    short = {'breach_home': 'dam-break\non own',
             'breach_away': 'dam-break\non rainfall',
             'harvey_home': 'rainfall\non own',
             'harvey_away': 'rainfall\non dam-break'}
    pos, labs = [], []
    for i, (key, colour, marker, filled, _) in enumerate(series):
        if key not in data:
            continue
        vols = [max(e['volume_ratio'], 1e-4) for e in data[key]['events']]
        p = axh.boxplot([vols], positions=[i], widths=0.6,
                        patch_artist=True, showfliers=False)
        for b in p['boxes']:
            b.set(facecolor=colour, alpha=0.65 if filled else 0.28,
                  edgecolor=colour, linewidth=1.5)
        for w in p['whiskers'] + p['caps']:
            w.set(color=colour, linewidth=1.3)
        for m in p['medians']:
            m.set(color=INK, linewidth=1.8)
        pos.append(i)
        labs.append(short[key])
    axh.set_xticks(pos)
    axh.set_xticklabels(labs, fontsize=8, rotation=90)
    axh.set_ylim(ax.get_ylim())
    axh.set_yticklabels([])
    axh.set_title('spread', fontsize=10, color=MUTED, loc='left')

    # four-row stats table, placed BELOW the axes so it can never overlap data
    for i, row in enumerate(rows):
        fig.text(0.06, -0.115 - i * 0.048, row, fontsize=9, color=INK,
                 family='monospace', ha='left', va='top')

    fig.text(0.5, -0.115 - len(rows) * 0.048 - 0.03,
             'One dot per held-out flood event. Pairwise disagreement = the share of '
             'event pairs where CSI and volume accuracy rank them in opposite order.',
             ha='center', fontsize=10, style='italic', color=MUTED)
    save(fig, 'figA_per_event_scatter')

    print("\nDisagreement summary used in the box:")
    for key, _, _, _, label in series:
        if key in data:
            d, t, pct = pairwise_disagreement(data[key]['events'])
            print(f"  {label:<32s} {pct:5.1f}%  ({d}/{t} pairs)")


if __name__ == '__main__':
    main()