"""
make_paper_figures.py
=====================

The paper's figure suite, built only from real result files.

Every number on every axis is read from a JSON in results/. Nothing is
typed in by hand. If a result file is missing, that figure is skipped
with a message rather than drawn from stale numbers.

Figures produced:
    figA_per_event_scatter   the lead figure - per-event CSI vs volume ratio
    figB_threshold_robust    relative vs fixed thresholds, incl. the reversal
    figC_mixing_regimes      five training regimes, breach vs harvey
    figD_mechanism_tag       does the flood-type tag help? CSI says no, volume says yes
    figE_finetune_curve      fine-tuning vs from-scratch, with the collapsed run flagged
    figF_rainfall_ablation   transfer with and without the rainfall channel

Run:
    python3 scripts/make_paper_figures.py

Outputs go to results/figures/ as both .png (200 dpi) and .pdf (vector).
"""

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')          # no screen needed on a server
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

RESULTS = 'results'
FIGDIR = 'results/figures'
os.makedirs(FIGDIR, exist_ok=True)


# ------------------------------------------------------------------ #
#  Shared house style
# ------------------------------------------------------------------ #

INK = '#1A2332'          # near-black for text
MUTED = '#6B7A8F'        # secondary text and gridlines
BREACH = '#1F6FB2'       # dam-break blue
HARVEY = '#C9522A'       # rainfall orange-red
GOOD = '#2E7D5B'         # green
WARN = '#B8860B'         # amber
DANGER = '#A62B2B'       # deep red
PAPER = '#FBFAF7'        # warm off-white background

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
    """Remove the top and right spines - cleaner look for a journal."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name):
    """Save both a raster and a vector copy."""
    for ext in ('png', 'pdf'):
        path = os.path.join(FIGDIR, f'{name}.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  saved {path}")
    plt.close(fig)


def load(filename):
    """Load a result file, or return None if it is not there yet."""
    path = os.path.join(RESULTS, filename)
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ #
#  FIGURE A - the lead figure: per-event CSI against volume ratio
# ------------------------------------------------------------------ #

def fig_per_event_scatter():
    """
    One dot per flood event. Horizontal axis is CSI (what the field
    reports). Vertical axis is volume ratio on a log scale (whether the
    amount of water is right at all).

    The shaded band is the honest zone - volume within a factor of two
    of the truth. Dots high above it are events where the model invented
    water. If those dots also sit far to the RIGHT, CSI called them good.
    """
    data = load('result_exp6_metric_disagreement.json')
    if data is None:
        return

    fig, (ax, axh) = plt.subplots(
        1, 2, figsize=(12.5, 6.4),
        gridspec_kw={'width_ratios': [3.4, 1], 'wspace': 0.06})

    style_axes(ax)
    style_axes(axh)

    series = [
        ('breach_home',  BREACH, 'o', 'full',  'Dam-break model, own data'),
        ('breach_away',  BREACH, 'o', 'none',  'Dam-break model, rainfall data'),
        ('harvey_home',  HARVEY, 's', 'full',  'Rainfall model, own data'),
        ('harvey_away',  HARVEY, 's', 'none',  'Rainfall model, dam-break data'),
    ]

    # the honest band: volume within a factor of two either way
    ax.axhspan(0.5, 2.0, color=GOOD, alpha=0.10, zorder=0)
    ax.axhline(1.0, color=GOOD, lw=1.4, ls='--', zorder=1)
    ax.text(0.015, 1.0, ' perfect water balance', va='bottom', ha='left',
            fontsize=9, color=GOOD, style='italic')

    all_vols = []
    for key, colour, marker, fill, label in series:
        if key not in data:
            continue
        events = data[key]['events']
        csis = [e['csi_final'] for e in events]
        vols = [max(e['volume_ratio'], 1e-4) for e in events]
        all_vols += vols
        ax.scatter(csis, vols,
                   marker=marker, s=95,
                   facecolors=colour if fill == 'full' else 'none',
                   edgecolors=colour, linewidths=1.8,
                   alpha=0.85 if fill == 'full' else 1.0,
                   zorder=4, label=label)

    # annotate the single worst disagreement in the paradox direction
    if 'breach_away' in data:
        ev = data['breach_away']['events']
        worst = max(ev, key=lambda e: e['volume_ratio'])
        ax.annotate(
            f"event {worst['event_index']}\n"
            f"CSI {worst['csi_final']:.2f} - looks fine\n"
            f"{worst['volume_ratio']:.0f}x too much water",
            xy=(worst['csi_final'], worst['volume_ratio']),
            xytext=(worst['csi_final'] - 0.30, worst['volume_ratio'] * 2.6),
            fontsize=9.5, color=DANGER, fontweight='bold', ha='left',
            arrowprops=dict(arrowstyle='->', color=DANGER, lw=1.6,
                            connectionstyle='arc3,rad=-0.25'),
            zorder=6)

    ax.set_yscale('log')
    ax.set_xlabel('CSI - the metric the field reports (higher looks better)',
                  fontsize=11.5)
    ax.set_ylabel('Volume ratio, log scale\n(predicted water / true water)',
                  fontsize=11.5)
    ax.set_xlim(0, 1.0)
    ax.legend(fontsize=9.5, loc='upper left', ncol=1)

    ax.set_title('Good CSI does not mean the water is right',
                 fontsize=14, fontweight='bold', loc='left', pad=14)

    # the correlation numbers, printed on the figure
    lines = []
    for key, _, _, _, label in series:
        if key in data:
            rho = data[key]['spearman_csi_vs_logvolerr']
            bad = data[key]['n_bad_disagreements']
            n = data[key]['n_events']
            lines.append(f"{label}:  rho = {rho:+.3f},  {bad}/{n} events disagree")
    ax.text(0.985, 0.03, '\n'.join(lines), transform=ax.transAxes,
            fontsize=8.8, ha='right', va='bottom', color=INK,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#D8DEE6'))

    # side panel: how far each group sits from a perfect volume ratio
    axh.set_yscale('log')
    axh.axhspan(0.5, 2.0, color=GOOD, alpha=0.10, zorder=0)
    axh.axhline(1.0, color=GOOD, lw=1.4, ls='--', zorder=1)
    positions, labels_short = [], []
    for i, (key, colour, marker, fill, label) in enumerate(series):
        if key not in data:
            continue
        vols = [max(e['volume_ratio'], 1e-4) for e in data[key]['events']]
        parts = axh.boxplot([vols], positions=[i], widths=0.62,
                            patch_artist=True, showfliers=False)
        for box in parts['boxes']:
            box.set(facecolor=colour, alpha=0.30 if fill == 'none' else 0.65,
                    edgecolor=colour, linewidth=1.5)
        for whisk in parts['whiskers'] + parts['caps']:
            whisk.set(color=colour, linewidth=1.3)
        for med in parts['medians']:
            med.set(color=INK, linewidth=1.8)
        positions.append(i)
        labels_short.append(label.replace(' model, ', '\n'))

    axh.set_xticks(positions)
    axh.set_xticklabels(labels_short, fontsize=7.6, rotation=0)
    axh.set_ylim(ax.get_ylim())
    axh.set_yticklabels([])
    axh.set_title('spread', fontsize=10, color=MUTED, loc='left')

    fig.text(0.5, -0.035,
             'One dot per held-out flood event. A negative correlation means the events '
             'CSI likes best are the events whose physics is most wrong.',
             ha='center', fontsize=10, style='italic', color=MUTED)

    save(fig, 'figA_per_event_scatter')


# ------------------------------------------------------------------ #
#  FIGURE B - threshold robustness, and where it breaks down
# ------------------------------------------------------------------ #

def fig_threshold_robustness():
    """
    Left: CSI at six RELATIVE thresholds (a percentage of each sample's
    own maximum depth). The paradox holds at every one.

    Right: CSI at FIXED absolute thresholds. Here the paradox disappears
    for the dam-break model and appears for the rainfall model instead.
    That is not a flaw to hide - it localises the cause to relative
    thresholding, which is the standard way to compare across datasets.
    """
    data = load('result_exp1_threshold_sweep.json')
    if data is None:
        return

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5.6),
                                   gridspec_kw={'width_ratios': [1.75, 1]})
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
                 label=f'{label} - own data')
        axl.plot(x, away, 'o--', color=colour, lw=2.4, ms=8,
                 markerfacecolor='white', markeredgewidth=2,
                 label=f'{label} - other flood type')
        # shade the paradox: wherever away sits ABOVE home
        axl.fill_between(x, home, away,
                         where=np.array(away) > np.array(home),
                         color=DANGER, alpha=0.13, zorder=0)

    axl.set_xscale('log')
    axl.set_xticks(x)
    axl.set_xticklabels([f'{v:g}%' for v in x])
    axl.set_xlabel('Wet/dry threshold, as a percentage of each sample\'s own '
                   'maximum depth', fontsize=11)
    axl.set_ylabel('CSI', fontsize=11.5)
    axl.set_title('Relative threshold: the paradox never closes',
                  fontsize=13, fontweight='bold', loc='left', pad=12)
    axl.legend(fontsize=9, loc='upper right')

    # mark the shaded region once
    axl.text(x[1], 0.70, 'shaded = model scores HIGHER\non the flood type it '
             'was never trained on',
             fontsize=9.5, color=DANGER, style='italic', va='top')

    # ---- right panel: fixed thresholds ----
    fixed = data['fixed']
    fths = sorted(fixed['breach'].keys(), key=float)
    width = 0.19
    idx = np.arange(len(fths) * 2, dtype=float)

    bars, labels = [], []
    pos = 0
    for src, colour, short in [('breach', BREACH, 'Dam-break'),
                               ('harvey', HARVEY, 'Rainfall')]:
        for t in fths:
            h = fixed[src][t]['home_csi']
            a = fixed[src][t]['away_csi']
            axr.bar(pos - width / 1.7, h, width, color=colour, alpha=0.95)
            axr.bar(pos + width / 1.7, a, width, color=colour, alpha=0.35,
                    edgecolor=colour, linewidth=1.4)
            if a > h:
                axr.annotate('paradox\nhere', xy=(pos + width / 1.7, a),
                             xytext=(pos + 0.05, a + 0.055),
                             fontsize=8.5, color=DANGER, fontweight='bold',
                             ha='center',
                             arrowprops=dict(arrowstyle='->', color=DANGER, lw=1.3))
            labels.append(f'{short}\n{float(t):g} m')
            pos += 1

    axr.set_xticks(range(len(labels)))
    axr.set_xticklabels(labels, fontsize=8.6)
    axr.set_ylabel('CSI', fontsize=11.5)
    axr.set_title('Fixed threshold: the paradox moves',
                  fontsize=13, fontweight='bold', loc='left', pad=12)
    axr.legend(handles=[
        Line2D([0], [0], color=MUTED, lw=8, alpha=0.95, label='own data'),
        Line2D([0], [0], color=MUTED, lw=8, alpha=0.35, label='other flood type'),
    ], fontsize=9, loc='upper right')

    fig.text(0.5, -0.04,
             'The paradox is a property of relative thresholding - the very practice '
             'used to make CSI comparable across datasets on different depth scales.',
             ha='center', fontsize=10, style='italic', color=MUTED)

    fig.tight_layout()
    save(fig, 'figB_threshold_robustness')


# ------------------------------------------------------------------ #
#  FIGURE C - five training regimes
# ------------------------------------------------------------------ #

def fig_mixing_regimes():
    """
    Does fixing the 17-to-1 sample imbalance rescue the small dataset?

    Each regime is one row. The two dots are its score on each test set,
    with the single-type baselines drawn as reference lines behind them.
    """
    data = load('result_exp7_balanced_mixed.json')
    if data is None:
        return

    order = ['breach_only', 'harvey_only', 'naive_mixed',
             'balanced_mixed', 'weighted_mixed']
    pretty = {'breach_only': 'Dam-break only',
              'harvey_only': 'Rainfall only',
              'naive_mixed': 'Mixed, as-is\n(17:1 imbalance)',
              'balanced_mixed': 'Mixed, balanced\n(equal draw)',
              'weighted_mixed': 'Mixed, loss-weighted\n(17.7x on dam-break)'}

    stats = {}
    for regime in order:
        rows = [r for r in data if r['regime'] == regime]
        if not rows:
            continue
        stats[regime] = {
            'breach': [r['breach_test']['csi_final'] for r in rows],
            'harvey': [r['harvey_test']['csi_final'] for r in rows],
        }

    fig, ax = plt.subplots(figsize=(11, 6.2))
    style_axes(ax)

    ys = np.arange(len(stats))[::-1].astype(float)

    # reference lines from the single-type baselines
    if 'breach_only' in stats:
        ref_b = np.median(stats['breach_only']['breach'])
        ax.axvline(ref_b, color=BREACH, ls=':', lw=1.8, alpha=0.8, zorder=1)
        ax.text(ref_b, len(stats) - 0.35, ' best dam-break score',
                color=BREACH, fontsize=9, style='italic', rotation=0)
    if 'harvey_only' in stats:
        ref_h = np.median(stats['harvey_only']['harvey'])
        ax.axvline(ref_h, color=HARVEY, ls=':', lw=1.8, alpha=0.8, zorder=1)
        ax.text(ref_h, len(stats) - 0.35, ' best rainfall score',
                color=HARVEY, fontsize=9, style='italic')

    for y, regime in zip(ys, [r for r in order if r in stats]):
        b = stats[regime]['breach']
        h = stats[regime]['harvey']
        bm, hm = np.median(b), np.median(h)

        # connector between the two medians
        ax.plot([bm, hm], [y, y], color='#C3CBD6', lw=2.5, zorder=2)
        # individual seeds, small and faint
        ax.scatter(b, [y] * len(b), s=34, color=BREACH, alpha=0.35, zorder=3)
        ax.scatter(h, [y] * len(h), s=34, color=HARVEY, alpha=0.35, zorder=3)
        # medians, large
        ax.scatter([bm], [y], s=190, color=BREACH, zorder=5,
                   edgecolor='white', linewidth=1.6)
        ax.scatter([hm], [y], s=190, color=HARVEY, zorder=5,
                   edgecolor='white', linewidth=1.6)
        ax.text(bm, y + 0.21, f'{bm:.3f}', ha='center', fontsize=9.5,
                color=BREACH, fontweight='bold')
        ax.text(hm, y + 0.21, f'{hm:.3f}', ha='center', fontsize=9.5,
                color=HARVEY, fontweight='bold')

    ax.set_yticks(ys)
    ax.set_yticklabels([pretty[r] for r in order if r in stats], fontsize=10.5)
    ax.set_xlabel('CSI on the held-out test set (median of 3 seeds, '
                  'faint dots are individual seeds)', fontsize=11)
    ax.set_title('No way of mixing the two flood types rescues the smaller one',
                 fontsize=14, fontweight='bold', loc='left', pad=14)
    ax.set_ylim(-0.7, len(stats) - 0.05)
    ax.grid(axis='y', visible=False)

    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=BREACH,
               markersize=12, label='tested on dam-break floods'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=HARVEY,
               markersize=12, label='tested on rainfall floods'),
    ], fontsize=10, loc='lower right')

    fig.text(0.5, -0.02,
             'Balanced sampling is the worst option for dam-break, not the best. '
             'The barrier is the flood mechanism, not the sample count.',
             ha='center', fontsize=10, style='italic', color=MUTED)

    fig.tight_layout()
    save(fig, 'figC_mixing_regimes')


# ------------------------------------------------------------------ #
#  FIGURE D - does telling the model the flood type help?
# ------------------------------------------------------------------ #

def fig_mechanism_tag():
    """
    Two panels side by side, both showing the same six runs.

    Left panel is CSI. Right panel is volume ratio. The arrows show what
    happens when the flood-type tag is switched on. They point in
    opposite directions, which is the whole point.
    """
    data = load('result_exp8_mechanism_aware.json')
    if data is None:
        return

    no = [r for r in data if not r['tagged']]
    yes = [r for r in data if r['tagged']]
    if not no or not yes:
        print("  [skip] exp8 file has no tagged/untagged split")
        return

    fig, (axc, axv) = plt.subplots(1, 2, figsize=(12.5, 6))
    style_axes(axc)
    style_axes(axv)

    def panel(ax, key_b, key_h, logscale, title, ylabel, perfect=None):
        groups = [('no tag', no, 0), ('with tag', yes, 1)]
        for name, rows, xpos in groups:
            for colour, key, off in [(BREACH, key_b, -0.11),
                                     (HARVEY, key_h, +0.11)]:
                vals = [r[key] for r in rows]
                med = np.median(vals)
                ax.scatter([xpos + off] * len(vals), vals, s=44,
                           color=colour, alpha=0.38, zorder=3)
                ax.scatter([xpos + off], [med], s=210, color=colour,
                           zorder=5, edgecolor='white', linewidth=1.8)
                ax.text(xpos + off, med, f'  {med:.2f}', fontsize=10,
                        color=colour, fontweight='bold', va='center')

        # arrows from no-tag median to with-tag median
        for colour, key, off in [(BREACH, key_b, -0.11), (HARVEY, key_h, +0.11)]:
            m0 = np.median([r[key] for r in no])
            m1 = np.median([r[key] for r in yes])
            ax.add_patch(FancyArrowPatch(
                (0 + off, m0), (1 + off, m1), arrowstyle='-|>',
                mutation_scale=20, color=colour, lw=2.2, alpha=0.75,
                connectionstyle='arc3,rad=0.12', zorder=4))

        if perfect is not None:
            ax.axhline(perfect, color=GOOD, ls='--', lw=1.5)
            ax.text(1.45, perfect, ' perfect', color=GOOD, fontsize=9,
                    style='italic', va='center')

        if logscale:
            ax.set_yscale('log')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['no tag', 'with flood-type tag'], fontsize=11)
        ax.set_xlim(-0.45, 1.55)
        ax.set_ylabel(ylabel, fontsize=11.5)
        ax.set_title(title, fontsize=13, fontweight='bold', loc='left', pad=12)

    panel(axc, 'breach_csi', 'harvey_csi', False,
          'What CSI says', 'CSI')
    panel(axv, 'breach_volume_ratio', 'harvey_volume_ratio', True,
          'What the water balance says',
          'Volume ratio, log scale', perfect=1.0)

    axc.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=BREACH,
               markersize=12, label='dam-break test set'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=HARVEY,
               markersize=12, label='rainfall test set'),
    ], fontsize=10, loc='lower left')

    fig.suptitle('The same change, judged two ways',
                 fontsize=15, fontweight='bold', color=INK, y=1.02, x=0.09,
                 ha='left')
    fig.text(0.5, -0.035,
             'Adding the flood-type tag moves CSI down on rainfall data and moves the '
             'water balance sharply towards correct. Judged on CSI alone, a real '
             'improvement would have been thrown away.',
             ha='center', fontsize=10, style='italic', color=MUTED)

    fig.tight_layout()
    save(fig, 'figD_mechanism_tag')


# ------------------------------------------------------------------ #
#  FIGURE E - fine-tuning against training from scratch
# ------------------------------------------------------------------ #

def fig_finetune_curve():
    """
    Does pre-training on a different flood type beat starting from
    nothing, once you have a few events of the real type?

    Medians are plotted, not means, because one run collapsed entirely
    and a mean would let that one run carry the conclusion. The collapsed
    run is drawn and labelled rather than removed.
    """
    data = load('result_exp4_finetune_curve.json')
    if data is None:
        return

    ns = sorted({r['n_events'] for r in data if r['n_events'] > 0})
    zero = [r['scores']['csi_final'] for r in data
            if r['n_events'] == 0 and r['mode'] == 'finetune']

    fig, ax = plt.subplots(figsize=(11, 6.3))
    style_axes(ax)

    curves = {}
    for mode, colour, label in [('finetune', BREACH, 'Pre-trained on the other flood type'),
                                ('scratch', WARN, 'Trained from scratch on the new type')]:
        med, lo, hi, seeds_by_n = [], [], [], []
        for n in ns:
            vals = [r['scores']['csi_final'] for r in data
                    if r['n_events'] == n and r['mode'] == mode]
            med.append(np.median(vals))
            lo.append(np.min(vals))
            hi.append(np.max(vals))
            seeds_by_n.append(vals)
        curves[mode] = (med, lo, hi, seeds_by_n)

        ax.fill_between(ns, lo, hi, color=colour, alpha=0.13, zorder=1)
        ax.plot(ns, med, 'o-', color=colour, lw=2.6, ms=9, zorder=4,
                label=label, markeredgecolor='white', markeredgewidth=1.5)
        for n, vals in zip(ns, seeds_by_n):
            ax.scatter([n] * len(vals), vals, s=28, color=colour,
                       alpha=0.40, zorder=3)

    # zero-shot reference
    if zero:
        z = np.median(zero)
        ax.axhline(z, color=MUTED, ls=':', lw=1.8)
        ax.text(ns[0], z, f'  no target data at all (median {z:.3f})',
                fontsize=9.5, color=MUTED, va='bottom', style='italic')

    # flag the collapsed run explicitly
    collapsed = [r for r in data
                 if r['scores']['csi_final'] < 0.02 and r['n_events'] > 0]
    for r in collapsed:
        ax.annotate(
            f"seed {r['seed']} collapsed\n(predicted almost no water:\n"
            f"volume ratio {r['scores']['volume_ratio']:.3f})",
            xy=(r['n_events'], r['scores']['csi_final']),
            xytext=(r['n_events'] * 0.42, 0.055),
            fontsize=9, color=DANGER, fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color=DANGER, lw=1.5,
                            connectionstyle='arc3,rad=0.2'))

    # shade where pre-training is actually ahead
    med_ft = curves['finetune'][0]
    med_sc = curves['scratch'][0]
    ax.fill_between(ns, med_ft, med_sc,
                    where=np.array(med_ft) > np.array(med_sc),
                    color=GOOD, alpha=0.18, zorder=0)

    ax.set_xscale('log')
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel('Number of independent events available from the new flood type',
                  fontsize=11.5)
    ax.set_ylabel('CSI on the new flood type', fontsize=11.5)
    ax.set_title('A head start under scarcity, not a general fix',
                 fontsize=14, fontweight='bold', loc='left', pad=14)
    ax.legend(fontsize=10.5, loc='lower right')
    ax.set_ylim(0, max(max(curves['finetune'][2]), max(curves['scratch'][2])) * 1.25)

    fig.text(0.5, -0.03,
             'Shaded green marks where pre-training beats starting from nothing. '
             'Lines are medians across three seeds; faint dots are the seeds themselves.',
             ha='center', fontsize=10, style='italic', color=MUTED)

    fig.tight_layout()
    save(fig, 'figE_finetune_curve')

    print("\n  Fine-tuning medians (CSI):")
    print(f"    {'events':>8s} {'pre-trained':>13s} {'from scratch':>13s}")
    for i, n in enumerate(ns):
        print(f"    {n:>8d} {curves['finetune'][0][i]:>13.4f} "
              f"{curves['scratch'][0][i]:>13.4f}")


# ------------------------------------------------------------------ #
#  FIGURE F - is the rainfall channel the explanation?
# ------------------------------------------------------------------ #

def fig_rainfall_ablation():
    """
    The obvious objection: dam-break data has no rainfall, so maybe the
    whole failure is just a missing input channel. This removes rainfall
    from BOTH datasets and repeats the transfer test.

    Drawn as a dumbbell chart: one line per seed, showing where the
    retained-skill figure moves when rainfall is taken away.
    """
    abl = load('result_exp2_rainfall_ablation.json')
    if abl is None:
        return

    # the with-rainfall numbers come from the original runs
    original = {}
    for path in glob.glob(os.path.join(RESULTS, 'result_*_vector_s*.json')):
        base = os.path.basename(path)
        if 'mixed' in base or 'finetune' in base:
            continue
        with open(path) as f:
            r = json.load(f)
        try:
            src = r['train_on']
            seed = r['seed'] if 'seed' in r else int(base.split('_s')[-1][0])
            home = r['home_scores']['csi_final']
            away = r['away_scores']['csi_final']
        except (KeyError, TypeError, ValueError):
            continue
        original[(src, seed)] = 100.0 * away / home if home > 0 else np.nan

    fig, ax = plt.subplots(figsize=(11, 6))
    style_axes(ax)

    rows = []
    for rec in abl:
        src, seed = rec['train_on'], rec['seed']
        after = rec['retained_pct']
        before = original.get((src, seed), np.nan)
        rows.append((src, seed, before, after))

    rows.sort(key=lambda t: (t[0], t[1]))
    ys = np.arange(len(rows))[::-1].astype(float)

    for y, (src, seed, before, after) in zip(ys, rows):
        colour = BREACH if src == 'breach' else HARVEY
        if np.isfinite(before):
            ax.plot([before, after], [y, y], color='#C3CBD6', lw=2.6, zorder=2)
            ax.scatter([before], [y], s=120, facecolors='white',
                       edgecolors=colour, linewidths=2.2, zorder=4)
        ax.scatter([after], [y], s=150, color=colour, zorder=5,
                   edgecolor='white', linewidth=1.5)
        ax.text(after, y + 0.26, f'{after:.0f}%', ha='center', fontsize=9.5,
                color=colour, fontweight='bold')

    ax.axvline(100, color=MUTED, ls='--', lw=1.6)
    ax.text(100, len(rows) - 0.4,
            ' 100% = no skill lost when the flood type changes',
            fontsize=9.5, color=MUTED, style='italic')

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{'Dam-break' if s == 'breach' else 'Rainfall'} "
                        f"trained, seed {sd}" for s, sd, _, _ in rows],
                       fontsize=10)
    ax.set_xlabel('Skill retained on the other flood type (%)', fontsize=11.5)
    ax.set_title('Removing the rainfall input changes nothing',
                 fontsize=14, fontweight='bold', loc='left', pad=14)
    ax.grid(axis='y', visible=False)

    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor=MUTED, markeredgewidth=2, markersize=11,
               label='with the rainfall channel'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MUTED,
               markersize=12, label='rainfall removed from both datasets'),
    ], fontsize=10, loc='lower right')

    fig.text(0.5, -0.02,
             'Values above 100% are the paradox, not a success: the model scores higher '
             'on a flood type it never saw, while predicting far too much water.',
             ha='center', fontsize=10, style='italic', color=MUTED)

    fig.tight_layout()
    save(fig, 'figF_rainfall_ablation')


# ------------------------------------------------------------------ #

def main():
    print("Building figures from real result files ...\n")
    for name, fn in [
        ('A  per-event scatter', fig_per_event_scatter),
        ('B  threshold robustness', fig_threshold_robustness),
        ('C  mixing regimes', fig_mixing_regimes),
        ('D  mechanism tag', fig_mechanism_tag),
        ('E  fine-tuning curve', fig_finetune_curve),
        ('F  rainfall ablation', fig_rainfall_ablation),
    ]:
        print(f"[{name}]")
        try:
            fn()
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
        print()

    print(f"Done. Figures in {FIGDIR}/")


if __name__ == '__main__':
    main()