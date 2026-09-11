"""
make_paper_figures3.py
======================

The controlled-experiment figure (exp 3 v2).

    figI_controlled     home skill against away skill, both mechanisms,
                        with the uninterpretable region shaded

Why this design: a "retained skill" percentage is a ratio, so it stops
meaning anything once the denominator (home skill) is near zero. Plotting
home against away instead of plotting the ratio makes that visible rather
than hiding it inside a number. One direction of this experiment is a
clean result; the other cannot be read at all, and the figure shows why.

Run:
    python3 scripts/make_paper_figures3.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS = 'results'
FIGDIR = 'results/figures'
os.makedirs(FIGDIR, exist_ok=True)

INK = '#1A2332'
MUTED = '#6B7A8F'
DIST = '#1F6FB2'        # distributed / rainfall-like forcing
POINT = '#C9522A'       # point-source forcing
GOOD = '#2E7D5B'
DANGER = '#A62B2B'
PAPER = '#FBFAF7'

# a model whose own-data score is below this cannot support a transfer
# claim, because the retained-skill ratio divides by it
HOME_FLOOR = 0.15

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


def fig_controlled():
    path = os.path.join(RESULTS, 'result_exp3_v2_stronger.json')
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return
    with open(path) as f:
        blob = json.load(f)

    cfg = blob.get('config', {})
    rows = blob['results']

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(13.5, 6.6), gridspec_kw={'width_ratios': [1.55, 1]})
    style_axes(ax)
    style_axes(axb)

    # ---------------- left: home skill vs away skill ----------------

    # the region where a transfer claim cannot be made
    ax.axvspan(0, HOME_FLOOR, color=DANGER, alpha=0.09, zorder=0)
    ax.text(HOME_FLOOR / 2, 1.03,
            'cannot be interpreted\nthe model failed at home,\n'
            'so "retained skill" divides by ~0',
            ha='center', va='top', fontsize=9.5, color=DANGER,
            fontweight='bold', style='italic')
    ax.axvline(HOME_FLOOR, color=DANGER, ls='--', lw=1.6, zorder=1)

    # the line where away equals home
    lim = 1.08
    ax.plot([0, lim], [0, lim], color=MUTED, ls=':', lw=1.6, zorder=1)
    ax.text(0.80, 0.83, 'away = home', rotation=39, fontsize=9,
            color=MUTED, style='italic')

    groups = [
        ('distributed', DIST, 'o',
         'Trained on distributed forcing\n(rainfall-like)'),
        ('point', POINT, 's',
         'Trained on point-source forcing\n(breach-like)'),
    ]

    for train_on, colour, marker, label in groups:
        sub = [r for r in rows if r['train_on'] == train_on]
        if not sub:
            continue
        hs = [r['home']['csi_final'] for r in sub]
        aws = [r['away_scores']['csi_final'] for r in sub]
        ax.scatter(hs, aws, marker=marker, s=175, color=colour,
                   edgecolor='white', linewidth=1.8, zorder=5, label=label)

        for r in sub:
            h = r['home']['csi_final']
            a = r['away_scores']['csi_final']
            vr = r['away_scores'].get('volume_ratio', float('nan'))
            degenerate = bool(r.get('degenerate', False))

            # label any point whose flood map looks perfect but whose
            # water volume is not
            if a > 0.9 and np.isfinite(vr) and (vr > 5 or vr < 0.2):
                ax.annotate(f"CSI {a:.2f}\nbut {vr:.0f}x the water",
                            xy=(h, a), xytext=(h + 0.10, a - 0.085),
                            fontsize=9, color=DANGER, fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color=DANGER,
                                            lw=1.4))
            if degenerate:
                ax.scatter([h], [a], marker='x', s=130, color=INK,
                           linewidth=2.2, zorder=6)

    ax.set_xlim(0, lim)
    ax.set_ylim(-0.04, lim)
    ax.set_xlabel('CSI on the forcing mechanism it was trained on',
                  fontsize=11.5)
    ax.set_ylabel('CSI on the other forcing mechanism', fontsize=11.5)
    ax.set_title('Everything held constant except the flood driver',
                 fontsize=14, fontweight='bold', loc='left', pad=14)
    ax.legend(fontsize=9.5, loc='center right')

    subtitle = (f"identical terrain seeds, "
                f"{cfg.get('dim', '?')}x{cfg.get('dim', '?')} grid, "
                f"same solver, {cfg.get('n_train', '?')} training runs per "
                f"mechanism, {cfg.get('epochs', '?')} epochs, "
                f"{len(cfg.get('seeds', []))} seeds")
    ax.text(0.0, 1.005, subtitle, transform=ax.transAxes, fontsize=9.5,
            color=MUTED, style='italic')

    # ---------------- right: the readable direction, per seed ----------

    sub = [r for r in rows if r['train_on'] == 'distributed']
    sub.sort(key=lambda r: r['seed'])
    if sub:
        ys = np.arange(len(sub))[::-1].astype(float)
        for y, r in zip(ys, sub):
            h = r['home']['csi_final']
            a = r['away_scores']['csi_final']
            axb.plot([a, h], [y, y], color='#C3CBD6', lw=2.6, zorder=2)
            axb.scatter([h], [y], s=150, color=DIST, zorder=4,
                        edgecolor='white', linewidth=1.5)
            axb.scatter([a], [y], s=150, facecolors='white',
                        edgecolors=DIST, linewidths=2.2, zorder=4)
            lost = 100.0 * (1 - a / h) if h > 0 else float('nan')
            axb.text(h + 0.035, y, f'{lost:.0f}% lost', fontsize=10,
                     color=INK, va='center', fontweight='bold')

        axb.set_yticks(ys)
        axb.set_yticklabels([f"seed {r['seed']}" for r in sub], fontsize=10.5)
        axb.set_xlim(0, 1.12)
        axb.set_xlabel('CSI', fontsize=11.5)
        axb.grid(axis='y', visible=False)

        med_lost = np.median([100.0 * (1 - r['away_scores']['csi_final'] /
                                       r['home']['csi_final'])
                              for r in sub if r['home']['csi_final'] > 0])
        axb.set_title('The readable direction: distributed  ->  point\n'
                      f'median {med_lost:.0f}% of skill lost',
                      fontsize=12.5, fontweight='bold', loc='left', pad=12)
        axb.legend(handles=[
            Line2D([0], [0], marker='o', color='w', markerfacecolor=DIST,
                   markersize=11, label='own mechanism'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor=DIST, markeredgewidth=2, markersize=11,
                   label='other mechanism'),
        ], fontsize=9.5, loc='lower right')

    fig.text(0.5, -0.035,
             'Only the blue points sit to the right of the red line, so only they '
             'support a transfer claim. The orange points never learned their own task, '
             'which is why their "retained skill" reaches four figures without meaning '
             'anything. Crosses mark runs flagged as degenerate by the experiment.',
             ha='center', fontsize=10, style='italic', color=MUTED)

    fig.tight_layout()
    save(fig, 'figI_controlled')

    # numbers for the paper text
    print("\n  Controlled experiment, per direction:")
    for train_on, _, _, _ in groups:
        s = [r for r in rows if r['train_on'] == train_on]
        if not s:
            continue
        hm = np.median([r['home']['csi_final'] for r in s])
        am = np.median([r['away_scores']['csi_final'] for r in s])
        rp = [r['retained_pct'] for r in s]
        flag = ' <-- below the home-skill floor, not interpretable' \
               if hm < HOME_FLOOR else ''
        print(f"    trained on {train_on:<12s} n={len(s)}  "
              f"home median {hm:.4f}  away median {am:.4f}  "
              f"retained median {np.median(rp):.1f}%{flag}")


if __name__ == '__main__':
    print("Building the controlled-experiment figure ...\n")
    try:
        fig_controlled()
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
    print(f"\nDone. Figures in {FIGDIR}/")