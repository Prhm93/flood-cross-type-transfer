"""
make_paper_figures_fixed.py
===========================

Regenerates the seven figures that had layout collisions or design
problems. Figures C and G were fine and are not touched.

    figA  annotation and box-plot labels moved out of the data area
    figB  explanatory text and the paradox marker moved inside the axes
    figD  value labels offset so they no longer sit on the legend
    figE  REDESIGNED - the min-max bands hid the medians. Now two panels:
          the medians on the left, and the difference between the two
          methods on the right, which is what the reader actually wants
    figF  title and reference label separated
    figH  REDESIGNED - the rescaled axis mixed three different quantities.
          Now a grid of small panels, one per metric, each on its own
          natural scale with an explicit verdict
    figI  the three stacked annotations spread out

Writes over the earlier PNG and PDF files of the same names.

Run:
    python3 scripts/make_paper_figures_fixed.py
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

INK = '#1A2332'
MUTED = '#6B7A8F'
BREACH = '#1F6FB2'
HARVEY = '#C9522A'
GOOD = '#2E7D5B'
WARN = '#B8860B'
DANGER = '#A62B2B'
PAPER = '#FBFAF7'
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


def load(filename):
    path = os.path.join(RESULTS, filename)
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ #
#  A - per-event scatter (layout fixed)
# ------------------------------------------------------------------ #

def figA():
    data = load('result_exp6_metric_disagreement.json')
    if data is None:
        return

    fig, (ax, axh) = plt.subplots(
        1, 2, figsize=(13.5, 7.2),
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
    ax.set_xlim(0, 1.0)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 6)          # headroom so the annotation fits inside

    ax.text(0.025, 1.15, 'perfect water balance', color=GOOD, fontsize=9.5,
            style='italic', va='bottom', transform=ax.get_yaxis_transform(),
            clip_on=False)

    if 'breach_away' in data:
        worst = max(data['breach_away']['events'],
                    key=lambda e: e['volume_ratio'])
        ax.annotate(
            f"event {worst['event_index']}:  CSI {worst['csi_final']:.2f} "
            f"looks acceptable,\nyet it predicts {worst['volume_ratio']:.0f} "
            f"times the true water",
            xy=(worst['csi_final'], worst['volume_ratio']),
            xytext=(0.46, hi * 2.6),
            fontsize=10, color=DANGER, fontweight='bold', ha='left',
            arrowprops=dict(arrowstyle='->', color=DANGER, lw=1.7,
                            connectionstyle='arc3,rad=0.28'), zorder=7)

    ax.set_xlabel('CSI - the metric the field reports (higher looks better)',
                  fontsize=11.5)
    ax.set_ylabel('Volume ratio, log scale\n(predicted water / true water)',
                  fontsize=11.5)
    ax.set_title('Good CSI does not mean the water is right',
                 fontsize=15, fontweight='bold', loc='left', pad=18)
    ax.legend(fontsize=9.5, loc='lower left', ncol=2,
              bbox_to_anchor=(0.0, 0.0))

    rows = []
    for key, _, _, _, label in series:
        if key in data:
            d = data[key]
            rows.append(f"{label}:  rho = {d['spearman_csi_vs_logvolerr']:+.3f}"
                        f",  {d['n_bad_disagreements']}/{d['n_events']} disagree")
    ax.text(0.985, 0.985, '\n'.join(rows), transform=ax.transAxes,
            fontsize=9, ha='right', va='top', color=INK,
            bbox=dict(boxstyle='round,pad=0.55', facecolor='white',
                      edgecolor='#D8DEE6'))

    # side panel - short labels, rotated, so they cannot run together
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

    fig.text(0.5, -0.045,
             'One dot per held-out flood event. A negative rank correlation means the '
             'events CSI scores highest are the events whose physics is most wrong.',
             ha='center', fontsize=10, style='italic', color=MUTED)
    save(fig, 'figA_per_event_scatter')


# ------------------------------------------------------------------ #
#  B - threshold robustness (layout fixed)
# ------------------------------------------------------------------ #

def figB():
    data = load('result_exp1_threshold_sweep.json')
    if data is None:
        return

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.5, 6.0),
                                   gridspec_kw={'width_ratios': [1.7, 1]})
    style_axes(axl)
    style_axes(axr)

    rel = data['relative']
    ths = sorted(rel['breach'].keys(), key=float)
    x = [float(t) * 100 for t in ths]

    for src, colour, label in [('breach', BREACH, 'Dam-break trained'),
                               ('harvey', HARVEY, 'Rainfall trained')]:
        home = [rel[src][t]['home_csi'] for t in ths]
        away = [rel[src][t]['away_csi'] for t in ths]
        axl.plot(x, home, 'o-', color=colour, lw=2.4, ms=8,
                 label=f'{label}, own data')
        axl.plot(x, away, 'o--', color=colour, lw=2.4, ms=8,
                 markerfacecolor='white', markeredgewidth=2,
                 label=f'{label}, other flood type')
        axl.fill_between(x, home, away,
                         where=np.array(away) > np.array(home),
                         color=DANGER, alpha=0.13, zorder=0)

    axl.set_xscale('log')
    axl.set_xticks(x)
    axl.set_xticklabels([f'{v:g}%' for v in x])
    axl.set_ylim(0, 0.99)
    axl.set_xlabel("Wet/dry threshold, as a percentage of each sample's own "
                   "maximum depth", fontsize=11)
    axl.set_ylabel('CSI', fontsize=11.5)
    axl.set_title('Relative threshold: the paradox never closes',
                  fontsize=13.5, fontweight='bold', loc='left', pad=12)
    axl.legend(fontsize=9, loc='lower left', ncol=2)
    axl.text(x[2], 0.93,
             'shaded band = the model scores HIGHER on the flood\n'
             'type it was never trained on',
             fontsize=9.5, color=DANGER, style='italic', va='top', ha='left')

    fixed = data['fixed']
    fths = sorted(fixed['breach'].keys(), key=float)
    width = 0.34
    labels, pos = [], 0
    ymax = 0
    for src, colour, short in [('breach', BREACH, 'Dam-break'),
                               ('harvey', HARVEY, 'Rainfall')]:
        for t in fths:
            h = fixed[src][t]['home_csi']
            a = fixed[src][t]['away_csi']
            ymax = max(ymax, h, a)
            axr.bar(pos - width / 1.9, h, width, color=colour, alpha=0.95)
            axr.bar(pos + width / 1.9, a, width, color=colour, alpha=0.32,
                    edgecolor=colour, linewidth=1.4)
            if a > h:
                axr.annotate('paradox appears\nin this direction',
                             xy=(pos + width / 1.9, a),
                             xytext=(pos - 0.30, a * 1.32),
                             fontsize=9, color=DANGER, fontweight='bold',
                             ha='center',
                             arrowprops=dict(arrowstyle='->', color=DANGER,
                                             lw=1.4))
            labels.append(f'{short}\n{float(t):g} m')
            pos += 1

    axr.set_ylim(0, ymax * 1.55)
    axr.set_xticks(range(len(labels)))
    axr.set_xticklabels(labels, fontsize=9)
    axr.set_ylabel('CSI', fontsize=11.5)
    axr.set_title('Fixed threshold: the paradox moves',
                  fontsize=13.5, fontweight='bold', loc='left', pad=12)
    axr.legend(handles=[
        Line2D([0], [0], color=MUTED, lw=9, alpha=0.95, label='own data'),
        Line2D([0], [0], color=MUTED, lw=9, alpha=0.32,
               label='other flood type')], fontsize=9, loc='upper left')

    fig.text(0.5, -0.05,
             'The paradox belongs to relative thresholding - the very practice used to '
             'make CSI comparable across datasets that sit on different depth scales.',
             ha='center', fontsize=10, style='italic', color=MUTED)
    fig.tight_layout()
    save(fig, 'figB_threshold_robustness')


# ------------------------------------------------------------------ #
#  D - mechanism tag (labels offset)
# ------------------------------------------------------------------ #

def figD():
    data = load('result_exp8_mechanism_aware.json')
    if data is None:
        return
    no = [r for r in data if not r['tagged']]
    yes = [r for r in data if r['tagged']]
    if not no or not yes:
        return

    fig, (axc, axv) = plt.subplots(1, 2, figsize=(13, 6.4))
    style_axes(axc)
    style_axes(axv)

    def panel(ax, kb, kh, logscale, title, ylabel, perfect=None):
        for name, rows, xpos in [('no', no, 0), ('yes', yes, 1)]:
            for colour, key, off in [(BREACH, kb, -0.13), (HARVEY, kh, +0.13)]:
                vals = [r[key] for r in rows]
                med = np.median(vals)
                ax.scatter([xpos + off] * len(vals), vals, s=44,
                           color=colour, alpha=0.35, zorder=3)
                ax.scatter([xpos + off], [med], s=215, color=colour, zorder=5,
                           edgecolor='white', linewidth=1.8)
                # label offset vertically, away from any legend
                ax.annotate(f'{med:.2f}', (xpos + off, med),
                            textcoords='offset points', xytext=(0, 15),
                            ha='center', fontsize=10.5, color=colour,
                            fontweight='bold', zorder=7)
        for colour, key, off in [(BREACH, kb, -0.13), (HARVEY, kh, +0.13)]:
            m0 = np.median([r[key] for r in no])
            m1 = np.median([r[key] for r in yes])
            ax.add_patch(FancyArrowPatch((0 + off, m0), (1 + off, m1),
                                         arrowstyle='-|>', mutation_scale=20,
                                         color=colour, lw=2.2, alpha=0.75,
                                         connectionstyle='arc3,rad=0.12',
                                         zorder=4))
        if perfect is not None:
            ax.axhline(perfect, color=GOOD, ls='--', lw=1.5)
            ax.annotate('perfect water balance', (0.02, perfect),
                        xycoords=('axes fraction', 'data'),
                        textcoords='offset points', xytext=(0, 6),
                        fontsize=9.5, color=GOOD, style='italic')
        if logscale:
            ax.set_yscale('log')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['no tag', 'with flood-type tag'], fontsize=11.5)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel(ylabel, fontsize=11.5)
        ax.set_title(title, fontsize=13.5, fontweight='bold', loc='left',
                     pad=12)

    panel(axc, 'breach_csi', 'harvey_csi', False, 'What CSI says', 'CSI')
    lo, hi = axc.get_ylim()
    axc.set_ylim(lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo))

    panel(axv, 'breach_volume_ratio', 'harvey_volume_ratio', True,
          'What the water balance says', 'Volume ratio, log scale',
          perfect=1.0)

    fig.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=BREACH,
               markersize=12, label='dam-break test set'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=HARVEY,
               markersize=12, label='rainfall test set')],
        fontsize=10.5, loc='upper right', bbox_to_anchor=(0.99, 1.02), ncol=2)

    fig.suptitle('The same change, judged two ways', fontsize=15.5,
                 fontweight='bold', color=INK, x=0.02, ha='left', y=1.03)
    fig.text(0.5, -0.04,
             'The tag moves CSI down on rainfall data while moving the water balance '
             'from 32x wrong to under 3x. Judged on CSI alone, a real improvement '
             'would have been discarded.',
             ha='center', fontsize=10, style='italic', color=MUTED)
    fig.tight_layout()
    save(fig, 'figD_mechanism_tag')


# ------------------------------------------------------------------ #
#  E - fine-tuning, REDESIGNED
# ------------------------------------------------------------------ #

def figE():
    """
    The old version drew min-max bands that swamped the medians.

    New design: medians only on the left (with the seeds as small dots),
    and on the right the thing the reader actually wants to know - how
    much pre-training gains or loses against starting from scratch, as a
    signed difference with a zero line.
    """
    data = load('result_exp4_finetune_curve.json')
    if data is None:
        return

    ns = sorted({r['n_events'] for r in data if r['n_events'] > 0})
    zero = [r['scores']['csi_final'] for r in data
            if r['n_events'] == 0 and r['mode'] == 'finetune']

    med = {}
    for mode in ('finetune', 'scratch'):
        med[mode] = [np.median([r['scores']['csi_final'] for r in data
                                if r['n_events'] == n and r['mode'] == mode])
                     for n in ns]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.5, 6.0),
                                   gridspec_kw={'width_ratios': [1.25, 1]})
    style_axes(axl)
    style_axes(axr)

    xs = np.arange(len(ns), dtype=float)

    for mode, colour, label in [
            ('finetune', BREACH, 'Pre-trained on the other flood type'),
            ('scratch', WARN, 'Trained from scratch on the new type')]:
        for i, n in enumerate(ns):
            vals = [r['scores']['csi_final'] for r in data
                    if r['n_events'] == n and r['mode'] == mode]
            axl.scatter([i] * len(vals), vals, s=30, color=colour,
                        alpha=0.35, zorder=3)
        axl.plot(xs, med[mode], 'o-', color=colour, lw=2.8, ms=10, zorder=5,
                 label=label, markeredgecolor='white', markeredgewidth=1.6)

    if zero:
        z = np.median(zero)
        axl.axhline(z, color=MUTED, ls=':', lw=1.8)
        axl.annotate(f'no target data at all (median {z:.3f})', (0, z),
                     textcoords='offset points', xytext=(4, 6),
                     fontsize=9.5, color=MUTED, style='italic')

    for r in [r for r in data if r['scores']['csi_final'] < 0.02
              and r['n_events'] > 0]:
        i = ns.index(r['n_events'])
        axl.annotate(f"seed {r['seed']} collapsed\n(volume ratio "
                     f"{r['scores']['volume_ratio']:.3f})",
                     xy=(i, r['scores']['csi_final']),
                     xytext=(i - 1.6, 0.045), fontsize=9, color=DANGER,
                     fontweight='bold', ha='left',
                     arrowprops=dict(arrowstyle='->', color=DANGER, lw=1.5,
                                     connectionstyle='arc3,rad=-0.25'))

    axl.set_xticks(xs)
    axl.set_xticklabels([str(n) for n in ns])
    axl.set_ylim(0, 0.27)
    axl.set_xlabel('Independent events available from the new flood type',
                   fontsize=11.5)
    axl.set_ylabel('CSI on the new flood type', fontsize=11.5)
    axl.set_title('Median across three seeds', fontsize=13,
                  fontweight='bold', loc='left', pad=12)
    axl.legend(fontsize=10, loc='lower right')

    # right panel: the signed difference
    diff = np.array(med['finetune']) - np.array(med['scratch'])
    colours = [GOOD if d > 0 else DANGER for d in diff]
    axr.bar(xs, diff, width=0.6, color=colours, alpha=0.88,
            edgecolor='white', linewidth=1.4)
    axr.axhline(0, color=INK, lw=1.6)
    for i, d in enumerate(diff):
        axr.annotate(f'{d:+.3f}', (i, d), textcoords='offset points',
                     xytext=(0, 7 if d > 0 else -16), ha='center',
                     fontsize=10, fontweight='bold',
                     color=GOOD if d > 0 else DANGER)

    span = max(abs(diff)) * 1.6
    axr.set_ylim(-span, span)
    axr.set_xticks(xs)
    axr.set_xticklabels([str(n) for n in ns])
    axr.set_xlabel('Independent events available from the new flood type',
                   fontsize=11.5)
    axr.set_ylabel('CSI gained by pre-training\n(above zero = pre-training wins)',
                   fontsize=11.5)
    axr.set_title('Where pre-training is worth having', fontsize=13,
                  fontweight='bold', loc='left', pad=12)

    cross = None
    for i in range(len(diff) - 1):
        if diff[i] > 0 >= diff[i + 1]:
            cross = (xs[i] + xs[i + 1]) / 2
    if cross is not None:
        axr.axvline(cross, color=MUTED, ls='--', lw=1.6)
        axr.annotate('crossover', (cross, span * 0.82),
                     textcoords='offset points', xytext=(6, 0),
                     fontsize=10, color=MUTED, style='italic')

    fig.suptitle('A head start under scarcity, not a general fix',
                 fontsize=15.5, fontweight='bold', color=INK, x=0.02,
                 ha='left', y=1.03)
    fig.text(0.5, -0.04,
             'Pre-training on a different flood type helps up to roughly ten '
             'independent events and reverses beyond twenty. One fine-tuned run at '
             '40 events collapsed entirely; medians are used so it cannot carry the '
             'conclusion on its own.',
             ha='center', fontsize=10, style='italic', color=MUTED)
    fig.tight_layout()
    save(fig, 'figE_finetune_curve')


# ------------------------------------------------------------------ #
#  F - rainfall ablation (title separated)
# ------------------------------------------------------------------ #

def figF():
    import glob
    abl = load('result_exp2_rainfall_ablation.json')
    if abl is None:
        return

    original = {}
    for path in glob.glob(os.path.join(RESULTS, 'result_*_vector_s*.json')):
        base = os.path.basename(path)
        if 'mixed' in base or 'finetune' in base:
            continue
        with open(path) as f:
            r = json.load(f)
        try:
            src = r['train_on']
            seed = r.get('seed', int(base.split('_s')[-1][0]))
            h = r['home_scores']['csi_final']
            a = r['away_scores']['csi_final']
        except (KeyError, TypeError, ValueError):
            continue
        original[(src, seed)] = 100.0 * a / h if h > 0 else np.nan

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    style_axes(ax)

    rows = sorted([(r['train_on'], r['seed'],
                    original.get((r['train_on'], r['seed']), np.nan),
                    r['retained_pct']) for r in abl],
                  key=lambda t: (t[0], t[1]))
    ys = np.arange(len(rows))[::-1].astype(float)

    for y, (src, seed, before, after) in zip(ys, rows):
        colour = BREACH if src == 'breach' else HARVEY
        if np.isfinite(before):
            ax.plot([before, after], [y, y], color='#C3CBD6', lw=2.8, zorder=2)
            ax.scatter([before], [y], s=125, facecolors='white',
                       edgecolors=colour, linewidths=2.2, zorder=4)
        ax.scatter([after], [y], s=155, color=colour, zorder=5,
                   edgecolor='white', linewidth=1.5)
        ax.annotate(f'{after:.0f}%', (after, y), textcoords='offset points',
                    xytext=(0, 14), ha='center', fontsize=10,
                    color=colour, fontweight='bold')

    ax.axvline(100, color=MUTED, ls='--', lw=1.7)
    ax.annotate('100% = no skill lost when the flood type changes',
                (100, -0.62), textcoords='offset points', xytext=(8, 0),
                fontsize=9.5, color=MUTED, style='italic', va='center')

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{'Dam-break' if s == 'breach' else 'Rainfall'} "
                        f"trained, seed {sd}" for s, sd, _, _ in rows],
                       fontsize=10.5)
    ax.set_ylim(-1.0, len(rows) - 0.3)
    ax.set_xlabel('Skill retained on the other flood type (%)', fontsize=11.5)
    ax.set_title('Removing the rainfall input changes nothing',
                 fontsize=15, fontweight='bold', loc='left', pad=18)
    ax.grid(axis='y', visible=False)
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor=MUTED, markeredgewidth=2, markersize=11,
               label='with the rainfall channel'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MUTED,
               markersize=12, label='rainfall removed from both datasets')],
        fontsize=10, loc='center right')

    fig.text(0.5, -0.02,
             'Values above 100% are the paradox, not a success: the model scores higher '
             'on a flood type it never saw, while predicting far too much water.',
             ha='center', fontsize=10, style='italic', color=MUTED)
    fig.tight_layout()
    save(fig, 'figF_rainfall_ablation')


# ------------------------------------------------------------------ #
#  H - three-way check, REDESIGNED
# ------------------------------------------------------------------ #

def figH():
    """
    The old version squeezed three different quantities onto one rescaled
    axis, which made the numbers unreadable. Each metric now gets its own
    small panel on its own natural scale, with an explicit verdict
    underneath.
    """
    data = load('result_exp1b_arrival_time.json')
    if data is None:
        return

    metrics = [
        ('CSI\nflood map', lambda s: s['csi_final'], '{:.3f}', False, True),
        ('Volume ratio\nwater balance',
         lambda s: s['mass_error'] + 1.0, '{:.1f}x', True, False),
        ('Arrival error\nhours', lambda s: s['arrival_mae_s'] / 3600.0,
         '{:.1f} h', False, False),
    ]
    # tuple: label, getter, format, log scale, higher-is-better

    models = [('breach', BREACH, 'Dam-break model tested on rainfall floods'),
              ('harvey', HARVEY, 'Rainfall model tested on dam-break floods')]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))

    for row, (src, colour, title) in enumerate(models):
        if src not in data:
            continue
        home, away = data[src]['home'], data[src]['away']

        for col, (name, get, fmt, logscale, higher_better) in enumerate(metrics):
            ax = axes[row, col]
            style_axes(ax)
            hv, av = get(home), get(away)

            if name.startswith('Volume'):
                better = abs(np.log10(max(av, 1e-6))) < abs(np.log10(max(hv, 1e-6)))
            elif higher_better:
                better = av > hv
            else:
                better = av < hv

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
            top = max(hv, av)
            if not logscale:
                ax.set_ylim(0, top * 1.34)

            verdict = 'moves the WRONG way' if better else 'moves the right way'
            vcolour = DANGER if better else GOOD
            ax.set_title(name, fontsize=11.5, fontweight='bold', pad=8)
            ax.text(0.5, -0.30, verdict, transform=ax.transAxes,
                    ha='center', fontsize=10.5, fontweight='bold',
                    color=vcolour)

        axes[row, 0].set_ylabel(title, fontsize=11, fontweight='bold',
                                color=colour, labelpad=12)

    fig.suptitle('Three independent checks on the same two models',
                 fontsize=15.5, fontweight='bold', color=INK, x=0.02,
                 ha='left', y=1.0)
    fig.text(0.5, -0.055,
             'Top row: CSI improves while both physical checks worsen - the strongest '
             'form of the disagreement. Bottom row: all three agree the transfer '
             'failed. "Moves the wrong way" means the metric flatters a model applied '
             'to a flood type it never saw. Single checkpoint (seed 0); not '
             'seed-replicated.',
             ha='center', fontsize=10, style='italic', color=MUTED)
    fig.tight_layout(h_pad=3.4)
    save(fig, 'figH_three_way')


# ------------------------------------------------------------------ #
#  I - controlled experiment (annotations spread out)
# ------------------------------------------------------------------ #

def figI():
    blob = load('result_exp3_v2_stronger.json')
    if blob is None:
        return
    cfg, rows = blob.get('config', {}), blob['results']

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(14, 6.8),
                                  gridspec_kw={'width_ratios': [1.5, 1]})
    style_axes(ax)
    style_axes(axb)

    ax.axvspan(0, HOME_FLOOR, color=DANGER, alpha=0.09, zorder=0)
    ax.axvline(HOME_FLOOR, color=DANGER, ls='--', lw=1.7, zorder=1)
    ax.plot([0, 1.1], [0, 1.1], color=MUTED, ls=':', lw=1.6, zorder=1)
    ax.annotate('away = home', (0.86, 0.86), rotation=38, fontsize=9.5,
                color=MUTED, style='italic', ha='center')

    # the warning sits low-left, inside the strip, well away from the points
    ax.text(HOME_FLOOR / 2, 0.50,
            'cannot be\ninterpreted:\nthe model failed\non its own task,\n'
            'so "retained skill"\ndivides by almost\nnothing',
            ha='center', va='center', fontsize=9, color=DANGER,
            fontweight='bold', style='italic', rotation=0)

    for train_on, colour, marker, label in [
            ('distributed', BREACH, 'o',
             'Trained on distributed forcing (rainfall-like)'),
            ('point', HARVEY, 's',
             'Trained on point-source forcing (breach-like)')]:
        sub = [r for r in rows if r['train_on'] == train_on]
        if not sub:
            continue
        ax.scatter([r['home']['csi_final'] for r in sub],
                   [r['away_scores']['csi_final'] for r in sub],
                   marker=marker, s=180, color=colour, edgecolor='white',
                   linewidth=1.8, zorder=5, label=label)

    # annotate the perfect-CSI cluster ONCE, to the right, not on top
    perfect = [r for r in rows if r['away_scores']['csi_final'] > 0.9]
    if perfect:
        vols = sorted(r['away_scores'].get('volume_ratio', np.nan)
                      for r in perfect)
        ax.annotate(
            f"{len(perfect)} runs scored a PERFECT flood map here\n"
            f"while predicting {vols[0]:.0f}x to {vols[-1]:.0f}x the true water",
            xy=(np.median([r['home']['csi_final'] for r in perfect]), 1.0),
            xytext=(0.36, 0.86), fontsize=10, color=DANGER,
            fontweight='bold', ha='left',
            arrowprops=dict(arrowstyle='->', color=DANGER, lw=1.7,
                            connectionstyle='arc3,rad=0.2'), zorder=7)

    ax.set_xlim(0, 1.1)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel('CSI on the forcing mechanism it was trained on',
                  fontsize=11.5)
    ax.set_ylabel('CSI on the other forcing mechanism', fontsize=11.5)
    ax.set_title('Everything held constant except the flood driver',
                 fontsize=14.5, fontweight='bold', loc='left', pad=20)
    ax.text(0.0, 1.012,
            f"identical terrain seeds, {cfg.get('dim','?')}x{cfg.get('dim','?')} "
            f"grid, same solver, {cfg.get('n_train','?')} training runs per "
            f"mechanism, {cfg.get('epochs','?')} epochs, "
            f"{len(cfg.get('seeds', []))} seeds",
            transform=ax.transAxes, fontsize=9.5, color=MUTED, style='italic')
    ax.legend(fontsize=10, loc='center right')

    sub = sorted([r for r in rows if r['train_on'] == 'distributed'],
                 key=lambda r: r['seed'])
    if sub:
        ys = np.arange(len(sub))[::-1].astype(float)
        for y, r in zip(ys, sub):
            h, a = r['home']['csi_final'], r['away_scores']['csi_final']
            axb.plot([a, h], [y, y], color='#C3CBD6', lw=2.8, zorder=2)
            axb.scatter([h], [y], s=155, color=BREACH, zorder=4,
                        edgecolor='white', linewidth=1.5)
            axb.scatter([a], [y], s=155, facecolors='white',
                        edgecolors=BREACH, linewidths=2.2, zorder=4)
            lost = 100.0 * (1 - a / h) if h > 0 else np.nan
            axb.annotate(f'{lost:.0f}% lost', (h, y),
                         textcoords='offset points', xytext=(12, 0),
                         fontsize=10.5, color=INK, va='center',
                         fontweight='bold')
        axb.set_yticks(ys)
        axb.set_yticklabels([f"seed {r['seed']}" for r in sub], fontsize=11)
        axb.set_ylim(-1.1, len(sub) - 0.4)
        axb.set_xlim(0, 1.22)
        axb.set_xlabel('CSI', fontsize=11.5)
        axb.grid(axis='y', visible=False)
        med = np.median([100.0 * (1 - r['away_scores']['csi_final'] /
                                  r['home']['csi_final'])
                         for r in sub if r['home']['csi_final'] > 0])
        axb.set_title('The readable direction: distributed to point\n'
                      f'median {med:.0f}% of skill lost',
                      fontsize=13, fontweight='bold', loc='left', pad=14)
        axb.legend(handles=[
            Line2D([0], [0], marker='o', color='w', markerfacecolor=BREACH,
                   markersize=11, label='own mechanism'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor=BREACH, markeredgewidth=2, markersize=11,
                   label='other mechanism')], fontsize=10, loc='lower center',
            ncol=2)

    fig.text(0.5, -0.045,
             'Only the blue points sit to the right of the red line, so only they can '
             'support a transfer claim. The orange points never learned their own task, '
             'which is why their retained-skill figures reach four digits without '
             'meaning anything.',
             ha='center', fontsize=10, style='italic', color=MUTED)
    fig.tight_layout()
    save(fig, 'figI_controlled')


def main():
    print("Regenerating the seven figures that needed fixing ...\n")
    for name, fn in [('A  per-event scatter', figA),
                     ('B  threshold robustness', figB),
                     ('D  mechanism tag', figD),
                     ('E  fine-tuning (redesigned)', figE),
                     ('F  rainfall ablation', figF),
                     ('H  three-way check (redesigned)', figH),
                     ('I  controlled experiment', figI)]:
        print(f"[{name}]")
        try:
            fn()
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
        print()
    print("C and G were already fine and were not touched.")


if __name__ == '__main__':
    main()