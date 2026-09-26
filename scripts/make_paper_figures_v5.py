#!/usr/bin/env python3
"""
make_paper_figures_v5.py - redraws the figures whose message changed in paper v5.
Every number comes from a result file. Saves over figures of the same name
(older versions stay in git history). Prints the numbers it plots.
  figA per-event scatter + all-wet CSI lines + pairwise table
  figB threshold sweep + all-wet CSI at every threshold (loads cache, ~1 min)
  figC mixing regimes + all-wet CSI lines
  figD mechanism tag, from the current exp8 file
  figE fine-tuning, no 'crossover' line
  figG capacity + all-wet CSI marks
  figH three metrics, seeds 0-2, verdicts from medians
  figI controlled experiment: CSI with all-wet lines + volume per seed
Run: python3 scripts/make_paper_figures_v5.py [--skip-B]
"""
import os, sys, json, math, itertools, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, 'results')
FIGDIR = os.path.join(RESULTS, 'figures')

INK, MUTED = '#1A2332', '#6B7A8F'
BREACH, HARVEY = '#1F6FB2', '#C9522A'
GOOD, WARN, DANGER, PAPER = '#2E7D5B', '#B8860B', '#A62B2B', '#FBFAF7'
S_POINT, S_DIST = '#7b3294', '#1b7837'      # synthetic, same as figJ/figK

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': MUTED,
    'axes.labelcolor': INK, 'axes.titlecolor': INK, 'axes.linewidth': 0.9,
    'axes.grid': True, 'grid.color': '#D8DEE6', 'grid.linewidth': 0.6, 'grid.alpha': 0.8,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'legend.frameon': False,
    'figure.facecolor': PAPER, 'axes.facecolor': PAPER, 'savefig.facecolor': PAPER})


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


def load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        print(f"  [skip] {p} not found")
        return None
    return json.load(open(p))


def pairwise(events):
    total, dis = 0, 0
    for a, b in itertools.combinations(events, 2):
        if a['csi_final'] == b['csi_final']:
            continue
        va = abs(math.log10(max(a['volume_ratio'], 1e-6)))
        vb = abs(math.log10(max(b['volume_ratio'], 1e-6)))
        if va == vb:
            continue
        total += 1
        dis += (a['csi_final'] > b['csi_final']) != (va < vb)
    return 100.0 * dis / total if total else float('nan')


# ------------------------------------------------------------------ A
def figA(AW):
    data, rec = load('result_exp6_metric_disagreement.json'), load('result_exp11_final_checks.json')
    if data is None:
        return
    series = [('breach_home', BREACH, 'o', True, 'Dam-break model, own data', 'breach->breach_s0'),
              ('breach_away', BREACH, 'o', False, 'Dam-break model, rainfall data', 'breach->harvey_s0'),
              ('harvey_home', HARVEY, 's', True, 'Rainfall model, own data', 'harvey->harvey_s0'),
              ('harvey_away', HARVEY, 's', False, 'Rainfall model, dam-break data', 'harvey->breach_s0')]
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    style_axes(ax)
    ax.axhspan(0.5, 2.0, color=GOOD, alpha=0.10, zorder=0)
    ax.axhline(1.0, color=GOOD, lw=1.4, ls='--', zorder=1)
    for key, col, mk, filled, lab, _ in series:
        if key in data:
            ev = data[key]['events']
            ax.scatter([e['csi_final'] for e in ev], [max(e['volume_ratio'], 1e-4) for e in ev],
                       marker=mk, s=95, facecolors=col if filled else 'none', edgecolors=col,
                       linewidths=1.8, alpha=0.85 if filled else 1.0, zorder=4, label=lab)
    ax.set_yscale('log')
    ax.set_xlim(0, 1.02)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 16)
    for te, col, nm in (('breach', BREACH, 'dam-break'), ('harvey', HARVEY, 'rainfall')):
        ax.axvline(AW[te], color=col, ls=':', lw=2.0, zorder=2)
        ax.text(AW[te] + 0.008, hi * 9, f'flooding every cell\nscores {AW[te]:.2f} on\n{nm} floods',
                color=col, fontsize=9, style='italic', va='top')
    ax.text(0.015, 1.13, 'perfect water balance', color=GOOD, fontsize=9.5, style='italic', va='bottom')
    if 'breach_away' in data:
        w = max(data['breach_away']['events'], key=lambda e: e['volume_ratio'])
        ax.annotate(f"event {w['event_index']}: CSI {w['csi_final']:.2f}\n{w['volume_ratio']:.0f} x the true water",
                    xy=(w['csi_final'], w['volume_ratio']), xytext=(0.40, hi * 2.5), fontsize=10,
                    color=DANGER, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=DANGER, lw=1.6), zorder=7)
    ax.set_xlabel('CSI (1% relative threshold)', fontsize=11.5)
    ax.set_ylabel('Volume ratio, log scale\n(predicted water / true water)', fontsize=11.5)
    ax.set_title('High CSI does not mean the water is right', fontsize=15, fontweight='bold', loc='left', pad=14)
    ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=4)
    print("\n  figA:")
    for i, (key, _, _, _, lab, rk) in enumerate(series):
        if key not in data:
            continue
        ev = data[key]['events']
        beat = rec['recount'][rk]['beat_allwet'] if rec and rk in rec.get('recount', {}) else '?'
        row = (f"{lab}: median CSI {np.median([e['csi_final'] for e in ev]):.3f}, "
               f"median volume {np.median([e['volume_ratio'] for e in ev]):.2f}x | "
               f"{pairwise(ev):.1f}% of pairs disagree | above-median CSI + bad volume "
               f"{data[key]['n_bad_disagreements']}/{data[key]['n_events']} | beat flooding everything {beat}/{len(ev)}")
        print("   ", row)
        fig.text(0.04, -0.12 - i * 0.045, row, fontsize=8.8, color=INK, family='monospace', va='top')
    save(fig, 'figA_per_event_scatter')


# ------------------------------------------------------------------ B
def truth_sets():
    import torch
    from src.experiment_lib import load_cache, build_model, get_device, rollout_predictions
    print("  loading cache for figB (about a minute) ...", flush=True)
    cache, dev = load_cache(), get_device()
    m = build_model('vector', 64, 3, dev)
    obj = torch.load(os.path.join(RESULTS, 'best_breach_vector_s0.pt'), map_location=dev, weights_only=False)
    if isinstance(obj, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    m.load_state_dict(obj)
    m.eval()
    with torch.no_grad():
        return {te: [t for _, t in rollout_predictions(m, cache[te]['test'], dev, 40, 19)]
                for te in ('breach', 'harvey')}


def allwet_at(trues, thr_fn):
    from src.metrics import csi
    return float(np.median([csi(np.full_like(t[-1], np.inf), t[-1], thr_fn(t)) for t in trues]))


def figB():
    data = load('result_exp1_threshold_sweep.json')
    if data is None:
        return
    T = truth_sets()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={'width_ratios': [1.75, 1]})
    style_axes(axl)
    style_axes(axr)
    rel = data['relative']
    ths = sorted(rel['breach'].keys(), key=float)
    x = [float(t) * 100 for t in ths]
    print("\n  figB all-wet CSI by relative threshold:")
    for te, col, nm in (('breach', BREACH, 'dam-break'), ('harvey', HARVEY, 'rainfall')):
        aw = [allwet_at(T[te], lambda t, f=float(th): max(float(np.max(t)) * f, 1e-9)) for th in ths]
        print(f"    {nm}: " + "  ".join(f"{xx:g}%={v:.3f}" for xx, v in zip(x, aw)))
        axl.plot(x, aw, 'x:', color=col, lw=2.0, ms=8, alpha=0.7, label=f'flooding every cell, {nm} floods')
    for src, col, lab in (('breach', BREACH, 'Dam-break trained'), ('harvey', HARVEY, 'Rainfall trained')):
        home = [rel[src][t]['home_csi'] for t in ths]
        away = [rel[src][t]['away_csi'] for t in ths]
        axl.plot(x, home, 'o-', color=col, lw=2.4, ms=8, label=f'{lab}, own data')
        axl.plot(x, away, 'o--', color=col, lw=2.4, ms=8, markerfacecolor='white',
                 markeredgewidth=2, label=f'{lab}, other flood type')
        axl.fill_between(x, home, away, where=np.array(away) > np.array(home), color=DANGER, alpha=0.10, zorder=0)
    axl.set_xscale('log')
    axl.set_xticks(x)
    axl.set_xticklabels([f'{v:g}%' for v in x])
    axl.set_xlabel("Wet/dry threshold, % of each sample's maximum depth", fontsize=11)
    axl.set_ylabel('CSI', fontsize=11.5)
    axl.set_title('Relative threshold', fontsize=13, fontweight='bold', loc='left', pad=12)
    axl.legend(fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)
    fixed = data['fixed']
    fths = sorted(fixed['breach'].keys(), key=float)
    width, pos, labels = 0.19, 0, []
    for src, col, short in (('breach', BREACH, 'Dam-break'), ('harvey', HARVEY, 'Rainfall')):
        other = 'harvey' if src == 'breach' else 'breach'
        for t in fths:
            h, a = fixed[src][t]['home_csi'], fixed[src][t]['away_csi']
            axr.bar(pos - width / 1.7, h, width, color=col, alpha=0.95)
            axr.bar(pos + width / 1.7, a, width, color=col, alpha=0.35, edgecolor=col, linewidth=1.4)
            for xx, te in ((pos - width / 1.7, src), (pos + width / 1.7, other)):
                axr.hlines(allwet_at(T[te], lambda tt, f=float(t): f), xx - width / 2, xx + width / 2,
                           colors=INK, lw=2.2, zorder=5)
            labels.append(f'{short}\n{float(t):g} m')
            pos += 1
    axr.set_xticks(range(len(labels)))
    axr.set_xticklabels(labels, fontsize=8.6)
    axr.set_ylabel('CSI', fontsize=11.5)
    axr.set_title('Fixed threshold', fontsize=13, fontweight='bold', loc='left', pad=12)
    axr.legend(handles=[Line2D([0], [0], color=MUTED, lw=8, alpha=0.95, label='own data'),
                        Line2D([0], [0], color=MUTED, lw=8, alpha=0.35, label='other flood type'),
                        Line2D([0], [0], color=INK, lw=2.2, label='flooding every cell')],
               fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    save(fig, 'figB_threshold_robustness')


# ------------------------------------------------------------------ C
def figC(AW):
    data = load('result_exp7_balanced_mixed.json')
    if data is None:
        return
    order = ['breach_only', 'harvey_only', 'naive_mixed', 'balanced_mixed', 'weighted_mixed']
    pretty = {'breach_only': 'Dam-break only', 'harvey_only': 'Rainfall only',
              'naive_mixed': 'Mixed, as-is\n(17:1 imbalance)', 'balanced_mixed': 'Mixed, balanced\n(equal draw)',
              'weighted_mixed': 'Mixed, loss-weighted\n(17.7x on dam-break)'}
    stats = {g: {'breach': [r['breach_test']['csi_final'] for r in data if r['regime'] == g],
                 'harvey': [r['harvey_test']['csi_final'] for r in data if r['regime'] == g]}
             for g in order if any(r['regime'] == g for r in data)}
    fig, ax = plt.subplots(figsize=(11, 6.4))
    style_axes(ax)
    for te, col in (('breach', BREACH), ('harvey', HARVEY)):
        ax.axvline(AW[te], color=col, ls='--', lw=1.6, alpha=0.8, zorder=1)
        ax.text(AW[te], -0.62, f' flooding every cell\n ({AW[te]:.2f})', color=col, fontsize=9, style='italic')
    ys = np.arange(len(stats))[::-1].astype(float)
    print("\n  figC medians:")
    for y, g in zip(ys, [g for g in order if g in stats]):
        b, h = stats[g]['breach'], stats[g]['harvey']
        bm, hm = np.median(b), np.median(h)
        print(f"    {g:15s} dam-break {bm:.3f}  rainfall {hm:.3f}")
        ax.plot([bm, hm], [y, y], color='#C3CBD6', lw=2.5, zorder=2)
        ax.scatter(b, [y] * len(b), s=34, color=BREACH, alpha=0.35, zorder=3)
        ax.scatter(h, [y] * len(h), s=34, color=HARVEY, alpha=0.35, zorder=3)
        ax.scatter([bm], [y], s=190, color=BREACH, zorder=5, edgecolor='white', linewidth=1.6)
        ax.scatter([hm], [y], s=190, color=HARVEY, zorder=5, edgecolor='white', linewidth=1.6)
        ax.text(bm, y + 0.21, f'{bm:.3f}', ha='center', fontsize=9.5, color=BREACH, fontweight='bold')
        ax.text(hm, y + 0.21, f'{hm:.3f}', ha='center', fontsize=9.5, color=HARVEY, fontweight='bold')
    ax.set_yticks(ys)
    ax.set_yticklabels([pretty[g] for g in order if g in stats], fontsize=10.5)
    ax.set_ylim(-0.8, len(stats) - 0.4)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('CSI on the held-out test set (median of 3 seeds; faint dots are seeds)', fontsize=11)
    ax.set_title('No way of mixing the two flood types rescues the smaller one',
                 fontsize=14, fontweight='bold', loc='left', pad=14)
    ax.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=BREACH, markersize=12, label='tested on dam-break floods'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor=HARVEY, markersize=12, label='tested on rainfall floods')],
              fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=2)
    save(fig, 'figC_mixing_regimes')


# ------------------------------------------------------------------ D
def figD():
    data = load('result_exp8_mechanism_aware.json')
    if data is None:
        return
    no, yes = [r for r in data if not r['tagged']], [r for r in data if r['tagged']]
    fig, (axc, axv) = plt.subplots(1, 2, figsize=(12.5, 6))
    med = {}
    for ax, kb, kh, logs, title in ((axc, 'breach_csi', 'harvey_csi', False, 'What CSI says'),
                                    (axv, 'breach_volume_ratio', 'harvey_volume_ratio', True, 'What the water balance says')):
        style_axes(ax)
        for name, rows, xp in (('no tag', no, 0), ('with tag', yes, 1)):
            for col, key, off in ((BREACH, kb, -0.11), (HARVEY, kh, 0.11)):
                vals = [r[key] for r in rows]
                m = float(np.median(vals))
                med[(key, name)] = (m, len(vals))
                ax.scatter([xp + off] * len(vals), vals, s=44, color=col, alpha=0.38, zorder=3)
                ax.scatter([xp + off], [m], s=210, color=col, zorder=5, edgecolor='white', linewidth=1.8)
                ax.text(xp + off, m, f'  {m:.2f}', fontsize=10, color=col, fontweight='bold', va='center')
        for col, key, off in ((BREACH, kb, -0.11), (HARVEY, kh, 0.11)):
            ax.add_patch(FancyArrowPatch((off, med[(key, 'no tag')][0]), (1 + off, med[(key, 'with tag')][0]),
                                         arrowstyle='-|>', mutation_scale=20, color=col, lw=2.2, alpha=0.75,
                                         connectionstyle='arc3,rad=0.12', zorder=4))
        if logs:
            ax.set_yscale('log')
            ax.axhline(1.0, color=GOOD, ls='--', lw=1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['no tag', 'with flood-type tag'], fontsize=11)
        ax.set_xlim(-0.45, 1.55)
        ax.set_title(title, fontsize=13, fontweight='bold', loc='left', pad=12)
    axc.set_ylabel('CSI', fontsize=11.5)
    axv.set_ylabel('Volume ratio, log scale', fontsize=11.5)
    axc.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=BREACH, markersize=12, label='dam-break test set'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor=HARVEY, markersize=12, label='rainfall test set')],
               fontsize=10, loc='upper center', bbox_to_anchor=(1.1, -0.08), ncol=2)
    fig.suptitle('The same change, judged two ways', fontsize=15, fontweight='bold', x=0.09, ha='left', y=1.02)
    print("\n  figD medians (compare with Table 17):")
    for k, (m, n) in sorted(med.items()):
        print(f"    {k[0]:22s} {k[1]:9s} median {m:.4f}  (n={n})")
    save(fig, 'figD_mechanism_tag')


# ------------------------------------------------------------------ E
def figE():
    data = load('result_exp4_finetune_curve.json')
    if data is None:
        return
    ns = sorted({r['n_events'] for r in data if r['n_events'] > 0})
    zero = [r['scores']['csi_final'] for r in data if r['n_events'] == 0 and r['mode'] == 'finetune']
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={'width_ratios': [1.4, 1]})
    style_axes(ax)
    style_axes(axd)
    meds = {}
    for mode, col, lab in (('finetune', BREACH, 'Pre-trained on the other flood type'),
                           ('scratch', WARN, 'Trained from scratch on the new type')):
        vals = [[r['scores']['csi_final'] for r in data if r['n_events'] == n and r['mode'] == mode] for n in ns]
        meds[mode] = [float(np.median(v)) for v in vals]
        ax.plot(ns, meds[mode], 'o-', color=col, lw=2.6, ms=9, zorder=4, label=lab,
                markeredgecolor='white', markeredgewidth=1.5)
        for n, v in zip(ns, vals):
            ax.scatter([n] * len(v), v, s=28, color=col, alpha=0.40, zorder=3)
    if zero:
        z = float(np.median(zero))
        ax.axhline(z, color=MUTED, ls=':', lw=1.8)
        ax.text(ns[0], z, f'  no target data (median {z:.3f})', fontsize=9, color=MUTED, va='bottom', style='italic')
    for r in [r for r in data if r['scores']['csi_final'] < 0.02 and r['n_events'] > 0]:
        ax.annotate(f"seed {r['seed']} collapsed\n(volume ratio {r['scores']['volume_ratio']:.3f})",
                    xy=(r['n_events'], r['scores']['csi_final']), xytext=(r['n_events'] * 0.35, 0.04),
                    fontsize=9, color=DANGER, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=DANGER, lw=1.4))
    ax.set_xscale('log')
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel('Independent events from the new flood type', fontsize=11)
    ax.set_ylabel('CSI on the new flood type', fontsize=11)
    ax.set_title('Median across three seeds', fontsize=12.5, fontweight='bold', loc='left', pad=10)
    ax.legend(fontsize=9.5, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2)
    diff = [a - b for a, b in zip(meds['finetune'], meds['scratch'])]
    axd.bar(range(len(ns)), diff, color=[GOOD if d > 0 else DANGER for d in diff], width=0.6)
    axd.axhline(0, color=INK, lw=1.2)
    for i, d in enumerate(diff):
        axd.text(i, d + (0.003 if d >= 0 else -0.003), f'{d:+.3f}', ha='center',
                 va='bottom' if d >= 0 else 'top', fontsize=9.5, fontweight='bold',
                 color=GOOD if d > 0 else DANGER)
    axd.set_xticks(range(len(ns)))
    axd.set_xticklabels([str(n) for n in ns])
    axd.set_xlabel('Independent events from the new flood type', fontsize=11)
    axd.set_ylabel('Pre-trained minus from scratch (CSI)', fontsize=11)
    axd.set_title('Difference in medians', fontsize=12.5, fontweight='bold', loc='left', pad=10)
    fig.suptitle('A possible head start under scarcity, not a general fix', fontsize=14,
                 fontweight='bold', x=0.06, ha='left', y=1.03)
    print("\n  figE differences:", [f'{n}: {d:+.3f}' for n, d in zip(ns, diff)])
    fig.tight_layout()
    save(fig, 'figE_finetune_curve')


# ------------------------------------------------------------------ G
def figG(AW):
    data = load('result_exp5_stronger_model.json')
    if data is None:
        return
    archs = sorted({r['arch'] for r in data}, key=lambda a: min(r['params'] for r in data if r['arch'] == a))
    params = {a: min(r['params'] for r in data if r['arch'] == a) for a in archs}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, tr, col, title in ((axes[0], 'harvey', HARVEY, 'Rainfall model tested on dam-break floods'),
                               (axes[1], 'breach', BREACH, 'Dam-break model tested on rainfall floods')):
        style_axes(ax)
        other = 'breach' if tr == 'harvey' else 'harvey'
        rows = [r for r in data if r['train_on'] == tr]
        for ai, arch in enumerate(archs):
            sub = [r for r in rows if r['arch'] == arch]
            xh, xa = ai * 1.4, ai * 1.4 + 0.72
            ax.hlines(AW[tr], xh - 0.2, xh + 0.2, colors=MUTED, linestyles='--', lw=2.2, zorder=1)
            ax.hlines(AW[other], xa - 0.2, xa + 0.2, colors=MUTED, linestyles='--', lw=2.2, zorder=1)
            for r in sub:
                h, a = r['home']['csi_final'], r['away']['csi_final']
                ax.plot([xh, xa], [h, a], color=col, lw=1.7, alpha=0.5, zorder=2)
                ax.scatter([xh], [h], s=52, color=col, alpha=0.75, zorder=3)
                ax.scatter([xa], [a], s=52, facecolors='white', edgecolors=col, linewidths=1.7, zorder=3)
            hm = np.median([r['home']['csi_final'] for r in sub])
            am = np.median([r['away']['csi_final'] for r in sub])
            ax.plot([xh, xa], [hm, am], color=INK, lw=3.0, zorder=5)
            ax.scatter([xh, xa], [hm, am], s=150, color=INK, zorder=6, edgecolor='white', linewidth=1.6)
            ax.text((xh + xa) / 2, -0.07, f'{arch}\n{params[arch]:,} parameters', ha='center',
                    fontsize=10, color=MUTED, transform=ax.get_xaxis_transform())
            print(f"  figG {tr} {arch}: home median {hm:.3f}  away median {am:.3f}")
        ax.set_xticks([])
        ax.set_xlim(-0.45, (len(archs) - 1) * 1.4 + 1.2)
        ax.set_ylim(0, 0.78)
        ax.set_ylabel('CSI', fontsize=11.5)
        ax.set_title(title, fontsize=12.5, fontweight='bold', loc='left', pad=12)
    fig.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=MUTED, markersize=10, label='own flood type'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor=MUTED, markeredgewidth=1.8, markersize=10, label='other flood type'),
                        Line2D([0], [0], color=INK, lw=3, label='median of 5 seeds'),
                        Line2D([0], [0], color=MUTED, lw=2.2, ls='--', label='flooding every cell')],
               loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=10)
    fig.suptitle('Eleven times the parameters, the same failure', fontsize=15, fontweight='bold', x=0.09, ha='left', y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, 'figG_capacity')


# ------------------------------------------------------------------ H
def verdict(k, h, a):
    if k == 'csi':
        return ('FLATTERS the transfer', DANGER) if a > h else ('reports the failure', GOOD)
    if k == 'volume_ratio':
        return (('reports the failure', GOOD) if abs(math.log(a)) > abs(math.log(h))
                else ('FLATTERS the transfer', DANGER))
    if a > 1.2 * h:
        return ('reports the failure', GOOD)
    if a < h / 1.2:
        return ('FLATTERS the transfer', DANGER)
    return ('inconclusive', MUTED)


def figH(AW):
    rec = load('result_exp11_final_checks.json')
    if rec is None:
        return
    t = rec['table16']
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
    rows = (('breach', BREACH, 'Dam-break model\ntested on rainfall floods'),
            ('harvey', HARVEY, 'Rainfall model\ntested on dam-break floods'))
    mets = (('csi', 'CSI\nflood map', False), ('volume_ratio', 'Volume ratio\nwater balance', True),
            ('arrival_h', 'Arrival error\nhours', False))
    print("\n  figH medians of seeds 0-2:")
    for ri, (tr, col, rlab) in enumerate(rows):
        other = 'harvey' if tr == 'breach' else 'breach'
        for ci, (k, name, logs) in enumerate(mets):
            ax = axes[ri, ci]
            style_axes(ax)
            own = [t[f'{tr}_s{s}_on_{tr}'][k] for s in range(3) if f'{tr}_s{s}_on_{tr}' in t]
            oth = [t[f'{tr}_s{s}_on_{other}'][k] for s in range(3) if f'{tr}_s{s}_on_{other}' in t]
            h, a = float(np.median(own)), float(np.median(oth))
            ax.bar([0], [h], 0.55, color=col, alpha=0.95)
            ax.bar([1], [a], 0.55, color=col, alpha=0.35, edgecolor=col, linewidth=1.4)
            ax.scatter([0] * len(own), own, s=24, color=INK, zorder=4)
            ax.scatter([1] * len(oth), oth, s=24, color=INK, zorder=4)
            fmt = (lambda v: f'{v:.3f}') if k == 'csi' else (lambda v: f'{v:.2f}x') if k == 'volume_ratio' else (lambda v: f'{v:.1f} h')
            for xx, v in ((0, h), (1, a)):
                ax.text(xx, v, fmt(v), ha='center', va='bottom', fontsize=10, fontweight='bold', color=INK)
            if k == 'csi':
                ax.hlines(AW[tr], -0.35, 0.35, colors=MUTED, linestyles='--', lw=2)
                ax.hlines(AW[other], 0.65, 1.35, colors=MUTED, linestyles='--', lw=2)
            if logs:
                ax.set_yscale('log')
                ax.axhline(1.0, color=GOOD, ls='--', lw=1.2)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['own flood\ntype', 'other flood\ntype'], fontsize=9)
            ax.set_title(name, fontsize=11, fontweight='bold')
            txt, c = verdict(k, h, a)
            ax.text(0.5, -0.32, txt, transform=ax.transAxes, ha='center', color=c, fontweight='bold', fontsize=10.5)
            print(f"    {tr:7s} {k:13s} own {h:.3f}  other {a:.3f}  -> {txt}")
        axes[ri, 0].set_ylabel(rlab, color=col, fontweight='bold', fontsize=10.5)
    fig.suptitle('No single metric reports the failure in both directions', fontsize=15, fontweight='bold', x=0.06, ha='left')
    fig.text(0.5, -0.03, 'Bars: medians of seeds 0-2; dots: individual seeds. Dashed grey on the CSI panels: '
             'CSI of a prediction that floods every cell.', ha='center', fontsize=10, style='italic', color=MUTED)
    fig.tight_layout(h_pad=3.5)
    save(fig, 'figH_three_way')


# ------------------------------------------------------------------ I
def figI():
    blob, s4 = load('result_exp3_v2_stronger.json'), load('result_exp9_section4_metrics.json')
    if blob is None:
        return
    rows, sat = blob['results'], (s4 or {}).get('saturation', {})
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(13.5, 6.6), gridspec_kw={'width_ratios': [1.4, 1]})
    style_axes(ax)
    style_axes(axb)
    ax.axvspan(0, 0.15, color=DANGER, alpha=0.09, zorder=0)
    ax.axvline(0.15, color=DANGER, ls='--', lw=1.6, zorder=1)
    ax.text(0.075, 0.55, 'below the\nhome-skill\nfloor', ha='center', fontsize=9.5, color=DANGER,
            fontweight='bold', style='italic')
    ax.plot([0, 1.08], [0, 1.08], color=MUTED, ls=':', lw=1.4, zorder=1)
    for mech, col in (('distributed', S_DIST), ('point', S_POINT)):
        if mech in sat:
            v = sat[mech]['median_allwet_csi']
            ax.axhline(v, color=col, ls='--', lw=1.4, alpha=0.8, zorder=1)
            ax.text(1.06, v + 0.012, f'flooding every cell scores {v:.2f} on {mech} floods',
                    color=col, fontsize=9, style='italic', ha='right')
    for mech, col, mk, lab in (('distributed', S_DIST, 'o', 'Trained on distributed forcing (rainfall-like)'),
                               ('point', S_POINT, 's', 'Trained on point-source forcing (breach-like)')):
        sub = [r for r in rows if r['train_on'] == mech]
        ax.scatter([r['home']['csi_final'] for r in sub], [r['away_scores']['csi_final'] for r in sub],
                   marker=mk, s=160, color=col, edgecolor='white', linewidth=1.6, zorder=5, label=lab)
    hi = [r for r in rows if r['train_on'] == 'point' and r['away_scores']['csi_final'] > 0.9]
    if hi:
        vr = [r['away_scores']['volume_ratio'] for r in hi]
        ax.annotate(f"CSI near 1 = what flooding every cell scores,\nwhile predicting {min(vr):.0f}-{max(vr):.0f}x the true water",
                    xy=(hi[0]['home']['csi_final'], hi[0]['away_scores']['csi_final']), xytext=(0.30, 0.72),
                    fontsize=9.5, color=DANGER, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=DANGER, lw=1.4))
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.04, 1.1)
    ax.set_xlabel('CSI on its own forcing mechanism', fontsize=11.5)
    ax.set_ylabel('CSI on the other forcing mechanism', fontsize=11.5)
    ax.set_title('Everything held constant except the flood driver', fontsize=13.5, fontweight='bold', loc='left', pad=12)
    ax.legend(fontsize=9.5, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1)
    sub = sorted([r for r in rows if r['train_on'] == 'distributed'], key=lambda r: r['seed'])
    ys = np.arange(len(sub))[::-1].astype(float)
    axb.axvspan(0.5, 2.0, color=GOOD, alpha=0.10, zorder=0)
    axb.axvline(1.0, color=GOOD, ls='--', lw=1.3)
    hv = [r['home']['volume_ratio'] for r in sub]
    av = [r['away_scores']['volume_ratio'] for r in sub]
    for y, h, a in zip(ys, hv, av):
        axb.plot([h, a], [y, y], color='#C3CBD6', lw=2.6, zorder=2)
        axb.scatter([h], [y], s=140, color=S_DIST, zorder=4, edgecolor='white', linewidth=1.5)
        axb.scatter([a], [y], s=140, facecolors='white', edgecolors=S_DIST, linewidths=2.2, zorder=4)
        axb.text(a * 1.15, y, f'{a:.1f}x', fontsize=10, color=INK, va='center', fontweight='bold')
    axb.set_xscale('log')
    axb.set_yticks(ys)
    axb.set_yticklabels([f"seed {r['seed']}" for r in sub])
    axb.grid(axis='y', visible=False)
    axb.set_xlabel('Volume ratio, log scale (predicted / true)', fontsize=11)
    axb.set_title(f'Distributed-trained model, volume ratio\n{np.median(hv):.2f}x at home, '
                  f'{np.median(av):.2f}x on point floods (medians)', fontsize=12, fontweight='bold', loc='left', pad=10)
    axb.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=S_DIST, markersize=11, label='own mechanism'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor=S_DIST, markeredgewidth=2, markersize=11, label='point floods')],
               fontsize=9.5, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    print(f"\n  figI distributed volume: home median {np.median(hv):.2f}x, away median {np.median(av):.2f}x")
    fig.tight_layout()
    save(fig, 'figI_controlled')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-B', action='store_true')
    a = ap.parse_args()
    sat = load('result_exp10_saturation_real.json')
    if sat is None:
        sys.exit('need result_exp10_saturation_real.json')
    AW = {k: sat['saturation'][k]['median_allwet_csi'] for k in ('breach', 'harvey')}
    print(f"all-wet CSI: dam-break {AW['breach']:.4f}, rainfall {AW['harvey']:.4f}")
    jobs = [('A', lambda: figA(AW)), ('C', lambda: figC(AW)), ('D', figD), ('E', figE),
            ('G', lambda: figG(AW)), ('H', lambda: figH(AW)), ('I', figI)]
    if not a.skip_B:
        jobs.insert(1, ('B', figB))
    for name, fn in jobs:
        print(f"\n[fig{name}]")
        try:
            fn()
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")


if __name__ == '__main__':
    main()
