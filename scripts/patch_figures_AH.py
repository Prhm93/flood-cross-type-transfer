"""
patch_figures_AH.py
===================

Two remaining fixes.

figA - the annotation ran underneath the correlation box and the legend
       sat on top of the orange squares. Legend now sits below the axes,
       the correlation box moves to the empty lower-right, and the
       annotation goes in the empty top-left.

figH - the caption was factually wrong. It claimed the bottom row showed
       all three metrics agreeing that the transfer failed. The figure
       itself shows the opposite: for the rainfall-trained model only CSI
       reports the failure, while volume ratio and arrival time both
       flatter it. The caption now says what the panels actually show.

Run:
    python3 scripts/patch_figures_AH.py
"""

import json
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


def load(fn):
    p = os.path.join(RESULTS, fn)
    if not os.path.exists(p):
        print(f"  [skip] {p} not found")
        return None
    with open(p) as f:
        return json.load(f)


# ------------------------------------------------------------------ #

def figA():
    data = load('result_exp6_metric_disagreement.json')
    if data is None:
        return

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
    ax.set_ylim(lo, hi * 16)         # room for the annotation above the cloud

    ax.text(0.015, 1.13, 'perfect water balance', color=GOOD, fontsize=9.5,
            style='italic', va='bottom')

    # annotation in the empty top-left
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

    # correlation box in the empty lower-right
    rows = []
    for key, _, _, _, label in series:
        if key in data:
            d = data[key]
            rows.append(f"{label}:  rho = {d['spearman_csi_vs_logvolerr']:+.3f}"
                        f",  {d['n_bad_disagreements']}/{d['n_events']} disagree")
    ax.text(0.985, 0.015, '\n'.join(rows), transform=ax.transAxes,
            fontsize=9, ha='right', va='bottom', color=INK,
            bbox=dict(boxstyle='round,pad=0.55', facecolor='white',
                      edgecolor='#D8DEE6'), zorder=8)

    ax.set_xlabel('CSI - the metric the field reports (higher looks better)',
                  fontsize=11.5)
    ax.set_ylabel('Volume ratio, log scale\n(predicted water / true water)',
                  fontsize=11.5)
    ax.set_title('Good CSI does not mean the water is right',
                 fontsize=15, fontweight='bold', loc='left', pad=16)
    # legend below the axes, clear of every data point
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

    fig.text(0.5, -0.10,
             'One dot per held-out flood event. A negative rank correlation means the '
             'events CSI scores highest are the events whose physics is most wrong.',
             ha='center', fontsize=10, style='italic', color=MUTED)
    save(fig, 'figA_per_event_scatter')


# ------------------------------------------------------------------ #

def figH():
    data = load('result_exp1b_arrival_time.json')
    if data is None:
        return

    metrics = [
        ('CSI\nflood map', lambda s: s['csi_final'], '{:.3f}', False, 'csi'),
        ('Volume ratio\nwater balance', lambda s: s['mass_error'] + 1.0,
         '{:.1f}x', True, 'vol'),
        ('Arrival error\nhours', lambda s: s['arrival_mae_s'] / 3600.0,
         '{:.1f} h', False, 'time'),
    ]
    models = [('breach', BREACH, 'Dam-break model tested on rainfall floods'),
              ('harvey', HARVEY, 'Rainfall model tested on dam-break floods')]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))

    for row, (src, colour, title) in enumerate(models):
        if src not in data:
            continue
        home, away = data[src]['home'], data[src]['away']

        for col, (name, get, fmt, logscale, kind) in enumerate(metrics):
            ax = axes[row, col]
            style_axes(ax)
            hv, av = get(home), get(away)

            if kind == 'csi':
                flatters = av > hv            # higher CSI away = flattering
            elif kind == 'vol':
                flatters = (abs(np.log10(max(av, 1e-6)))
                            < abs(np.log10(max(hv, 1e-6))))
            else:
                flatters = av < hv            # lower error away = flattering

            ax.bar([0], [hv], width=0.55, color=colour, alpha=0.92)
            ax.bar([1], [av], width=0.55, color=colour, alpha=0.30,
                   edgecolor=colour, linewidth=1.8)
            if logscale:
                ax.set_yscale('log')
                ax.axhline(1.0, color=GOOD, ls='--', lw=1.4)

            for xp, v in [(0, hv), (1, av)]:
                ax.annotate(fmt.format(v), (xp, v),
                            textcoords='offset points', xytext=(0, 6),
                            ha='center', fontsize=11, fontweight='bold',
                            color=INK)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['own flood\ntype', 'other flood\ntype'],
                               fontsize=9.5)
            ax.set_xlim(-0.6, 1.6)
            if not logscale:
                ax.set_ylim(0, max(hv, av) * 1.34)

            ax.set_title(name, fontsize=11.5, fontweight='bold', pad=8)
            ax.text(0.5, -0.30,
                    'FLATTERS the transfer' if flatters
                    else 'reports the failure',
                    transform=ax.transAxes, ha='center', fontsize=10.5,
                    fontweight='bold', color=DANGER if flatters else GOOD)

        axes[row, 0].set_ylabel(title, fontsize=11, fontweight='bold',
                                color=colour, labelpad=12)

    fig.suptitle('No single metric reports the failure in both directions',
                 fontsize=15.5, fontweight='bold', color=INK, x=0.02,
                 ha='left', y=1.0)
    fig.text(0.5, -0.075,
             'Top row: CSI flatters a dam-break model applied to rainfall floods, while '
             'both physical checks catch it. Bottom row: the pattern reverses - CSI '
             'correctly reports the failure, but the rainfall-trained model is so wet on '
             'its own data (12.5x) that both the water balance and the timing look '
             'better on the flood type it never saw. Reporting any one of these three '
             'alone would give the wrong answer in one direction or the other. '
             'Single checkpoint (seed 0); not seed-replicated.',
             ha='center', fontsize=10, style='italic', color=MUTED)
    fig.tight_layout(h_pad=3.4)
    save(fig, 'figH_three_way')


if __name__ == '__main__':
    print("Patching figures A and H ...\n")
    for name, fn in [('A', figA), ('H', figH)]:
        print(f"[{name}]")
        try:
            fn()
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
        print()
    print("Done.")