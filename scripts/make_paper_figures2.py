"""
make_paper_figures2.py
======================

The remaining figures, in the same house style as make_paper_figures.py.

    figG_capacity        does a bigger model transfer better? (exp 5, 5 seeds)
    figH_three_way       CSI vs water balance vs timing, all at once (exp 1b)

Run:
    python3 scripts/make_paper_figures2.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

RESULTS = 'results'
FIGDIR = 'results/figures'
os.makedirs(FIGDIR, exist_ok=True)

# same palette as the first script
INK = '#1A2332'
MUTED = '#6B7A8F'
BREACH = '#1F6FB2'
HARVEY = '#C9522A'
GOOD = '#2E7D5B'
WARN = '#B8860B'
DANGER = '#A62B2B'
PAPER = '#FBFAF7'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.edgecolor': MUTED,
    'axes.labelcolor': INK,
    'axes.titlecolor': INK,
    'axes.linewidth': 0.9,
    'axes.grid': True,
    'grid.color': '#D8DEE6',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.8,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'legend.frameon': False,
    'figure.facecolor': PAPER,
    'axes.facecolor': PAPER,
    'savefig.facecolor': PAPER,
})


def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ('png', 'pdf'):
        path = os.path.join(FIGDIR, f'{name}.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  saved {path}")
    plt.close(fig)


def load(filename):
    path = os.path.join(RESULTS, filename)
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ #
#  FIGURE G - does more model capacity close the gap?
# ------------------------------------------------------------------ #

def fig_capacity():
    """
    Each seed is one line running from its home score to its away score.
    A steep drop means the model did not survive the change of flood type.

    Two panels because the two directions behave completely differently:
    one loses skill, the other gains it (which is the paradox, not a win).
    """
    data = load('result_exp5_stronger_model.json')
    if data is None:
        return

    archs = sorted({r['arch'] for r in data},
                   key=lambda a: min(r['params'] for r in data if r['arch'] == a))
    params = {a: min(r['params'] for r in data if r['arch'] == a) for a in archs}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))

    for ax, train_on, colour, title in [
            (axes[0], 'harvey', HARVEY,
             'Rainfall model tested on dam-break floods'),
            (axes[1], 'breach', BREACH,
             'Dam-break model tested on rainfall floods')]:

        style_axes(ax)
        rows = [r for r in data if r['train_on'] == train_on]
        if not rows:
            continue

        for ai, arch in enumerate(archs):
            sub = [r for r in rows if r['arch'] == arch]
            x_home = ai * 1.4
            x_away = ai * 1.4 + 0.72

            for r in sub:
                h = r['home']['csi_final']
                a = r['away']['csi_final']
                drop = a < h
                ax.plot([x_home, x_away], [h, a],
                        color=DANGER if not drop else colour,
                        lw=1.7, alpha=0.55, zorder=2)
                ax.scatter([x_home], [h], s=52, color=colour, alpha=0.75,
                           zorder=3)
                ax.scatter([x_away], [a], s=52,
                           facecolors='white', edgecolors=colour,
                           linewidths=1.7, zorder=3)

            hm = np.median([r['home']['csi_final'] for r in sub])
            am = np.median([r['away']['csi_final'] for r in sub])
            ret = np.median([r['away']['csi_final'] / r['home']['csi_final']
                             for r in sub if r['home']['csi_final'] > 0]) * 100

            ax.plot([x_home, x_away], [hm, am], color=INK, lw=3.0, zorder=5)
            ax.scatter([x_home, x_away], [hm, am], s=150, color=INK, zorder=6,
                       edgecolor='white', linewidth=1.6)

            mid = (x_home + x_away) / 2
            ax.text(mid, max(hm, am) + 0.045,
                    f'{ret:.0f}% retained',
                    ha='center', fontsize=11, fontweight='bold',
                    color=DANGER if ret > 100 else INK)
            ax.text(mid, -0.055,
                    f'{arch}\n{params[arch]:,} parameters',
                    ha='center', fontsize=10, color=MUTED,
                    transform=ax.get_xaxis_transform())

        ax.set_xticks([])
        ax.set_xlim(-0.45, (len(archs) - 1) * 1.4 + 1.2)
        ax.set_ylabel('CSI', fontsize=11.5)
        ax.set_title(title, fontsize=12.5, fontweight='bold', loc='left',
                     pad=12)
        ax.set_ylim(0, 0.78)

    axes[0].legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=HARVEY,
               markersize=10, label='own flood type'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor=HARVEY, markeredgewidth=1.8, markersize=10,
               label='other flood type'),
        Line2D([0], [0], color=INK, lw=3, label='median of 5 seeds'),
    ], fontsize=9.5, loc='upper right')

    fig.suptitle('Eleven times the parameters, the same failure',
                 fontsize=15, fontweight='bold', color=INK,
                 x=0.09, ha='left', y=1.02)

    fig.text(0.5, -0.06,
             'Left: the bigger model retains LESS, not more. Right: scores above 100% '
             'are the paradox - the model appears to improve on a flood type it never '
             'saw, while its water balance is wrong by two orders of magnitude.',
             ha='center', fontsize=10, style='italic', color=MUTED)

    fig.tight_layout()
    save(fig, 'figG_capacity')

    print("\n  Capacity medians (retained %):")
    for arch in archs:
        for train_on in ('breach', 'harvey'):
            sub = [r for r in data
                   if r['arch'] == arch and r['train_on'] == train_on]
            if not sub:
                continue
            ret = np.median([r['away']['csi_final'] / r['home']['csi_final']
                             for r in sub if r['home']['csi_final'] > 0]) * 100
            print(f"    {arch:>6s} trained on {train_on:<7s} "
                  f"n={len(sub)}  median {ret:7.1f}%")


# ------------------------------------------------------------------ #
#  FIGURE H - three metrics, one picture
# ------------------------------------------------------------------ #

def fig_three_way():
    """
    Three independent checks on the same two models, drawn on a shared
    scale so they can be compared at a glance.

    Each metric is redrawn so that UP always means WORSE. If CSI points
    down (better) while the other two point up (worse), the metric is
    contradicting the physics.
    """
    data = load('result_exp1b_arrival_time.json')
    if data is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True)

    metrics = [
        ('CSI\n(flood map)', lambda s: 1.0 - s['csi_final'],
         lambda s: f"{s['csi_final']:.3f}"),
        ('Water balance\n(volume ratio)',
         lambda s: abs(np.log10(max(s['mass_error'] + 1.0, 1e-6))),
         lambda s: f"{s['mass_error'] + 1.0:.1f}x"),
        ('Arrival timing\n(hours of error)',
         lambda s: s['arrival_mae_s'] / 3600.0 / 24.0,
         lambda s: f"{s['arrival_mae_s'] / 3600.0:.1f} h"),
    ]

    for ax, src, colour, title in [
            (axes[0], 'breach', BREACH,
             'Dam-break model  ->  tested on rainfall floods'),
            (axes[1], 'harvey', HARVEY,
             'Rainfall model  ->  tested on dam-break floods')]:

        style_axes(ax)
        if src not in data:
            continue
        home, away = data[src]['home'], data[src]['away']

        xs = np.arange(len(metrics), dtype=float)
        home_v = [m[1](home) for m in metrics]
        away_v = [m[1](away) for m in metrics]

        ax.plot(xs, home_v, 'o-', color=colour, lw=2.6, ms=13, zorder=4,
                markeredgecolor='white', markeredgewidth=1.8,
                label='on its own flood type')
        ax.plot(xs, away_v, 'o--', color=colour, lw=2.6, ms=13, zorder=4,
                markerfacecolor='white', markeredgewidth=2.4,
                label='on the other flood type')

        for i, (name, fn, fmt) in enumerate(metrics):
            hv, av = home_v[i], away_v[i]
            worse = av > hv
            # arrow showing which way the metric moved
            ax.add_patch(FancyArrowPatch(
                (i + 0.18, hv), (i + 0.18, av), arrowstyle='-|>',
                mutation_scale=17,
                color=DANGER if worse else GOOD, lw=2.0, alpha=0.85, zorder=3))
            ax.text(i + 0.26, (hv + av) / 2,
                    'worse' if worse else 'looks\nbetter',
                    fontsize=9, fontweight='bold', va='center',
                    color=DANGER if worse else GOOD)
            ax.text(i, hv - 0.055, fmt(home), ha='center', fontsize=9.5,
                    color=colour, fontweight='bold')
            ax.text(i, av + 0.035, fmt(away), ha='center', fontsize=9.5,
                    color=colour, fontweight='bold')

        ax.set_xticks(xs)
        ax.set_xticklabels([m[0] for m in metrics], fontsize=10.5)
        ax.set_xlim(-0.45, len(metrics) - 0.2)
        ax.set_title(title, fontsize=12.5, fontweight='bold', loc='left',
                     pad=12)
        ax.legend(fontsize=9.5, loc='upper left')

    axes[0].set_ylabel('rescaled so that UP always means WORSE\n'
                       '(each metric on its own natural scale)',
                       fontsize=10.5)

    fig.suptitle('Three checks on the same model, and only one of them '
                 'is reassuring',
                 fontsize=15, fontweight='bold', color=INK, x=0.09,
                 ha='left', y=1.02)

    fig.text(0.5, -0.04,
             'Left: CSI moves the wrong way while both physical checks move the right '
             'way - the strongest form of the disagreement. Right: no disagreement, all '
             'three agree the transfer failed. Single seed (s0); not seed-replicated.',
             ha='center', fontsize=10, style='italic', color=MUTED)

    fig.tight_layout()
    save(fig, 'figH_three_way')


def main():
    print("Building the remaining figures ...\n")
    for name, fn in [('G  capacity', fig_capacity),
                     ('H  three-way check', fig_three_way)]:
        print(f"[{name}]")
        try:
            fn()
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
        print()
    print(f"Done. Figures in {FIGDIR}/")


if __name__ == '__main__':
    main()