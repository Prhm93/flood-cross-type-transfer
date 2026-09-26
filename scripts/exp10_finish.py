#!/usr/bin/env python3
"""
exp10_finish.py - last checks and figures for the extension. No training.
A. Saturation on the REAL test sets (breach, Harvey): share of cells truly wet at the
   final step (1% auto threshold) and CSI of a trivial 'everything wet' prediction,
   next to the trained breach and Harvey models (seeds 0-2, 40 steps, 19 events).
B. figJ: 3 x 3 synthetic matrix, CSI and volume ratio, trivial all-wet CSI marked.
C. figK: saturation across all five test sets.
Never overwrites results or figures.
"""
import os, sys, ast, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.experiment_lib import load_cache, build_model, get_device, rollout_predictions, evaluate
from src.metrics import auto_threshold, csi

RESULTS = os.path.join(ROOT, 'results')
FIG = os.path.join(RESULTS, 'figures')
OUT_JSON = os.path.join(RESULTS, 'result_exp10_saturation_real.json')
SEC4 = os.path.join(RESULTS, 'result_exp9_section4_metrics.json')
REAL, SYN = ['breach', 'harvey'], ['point', 'distributed', 'inflow']


def load_ckpt(model, path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    model.load_state_dict(obj)


def house_style():
    """Collect colour constants ('#...') defined at top level in the existing figure scripts."""
    found = {}
    for fn in ['make_paper_figures_fixed.py', 'patch_figures_AH.py', 'make_paper_figures3.py',
               'make_paper_figures2.py', 'make_paper_figures.py']:
        p = os.path.join(ROOT, 'scripts', fn)
        if not os.path.exists(p):
            continue
        for node in ast.parse(open(p).read()).body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str) and node.value.value.startswith('#')):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id not in found:
                        found[t.id] = node.value.value
    return found


def part_a():
    if os.path.exists(OUT_JSON):
        print(f"[A] {OUT_JSON} exists - loading it instead of recomputing")
        return json.load(open(OUT_JSON))
    print("[A] loading cache (about a minute) ...", flush=True)
    cache = load_cache()
    device = get_device()
    ck = lambda tr, s: os.path.join(RESULTS, f'best_{tr}_vector_s{s}.pt')
    real = {'saturation': {}, 'models': []}
    ref = build_model('vector', 64, 3, device)
    load_ckpt(ref, ck('breach', 0), device)
    ref.eval()
    with torch.no_grad():
        for te in REAL:
            roll = rollout_predictions(ref, cache[te]['test'], device, 40, 19)
            wet, allwet = [], []
            for pred, true in roll:
                thr = auto_threshold(true[:, np.newaxis, :])
                wet.append(float((true[-1] > thr).mean()))
                allwet.append(float(csi(np.full_like(true[-1], np.inf), true[-1], thr)))
            real['saturation'][te] = {'median_true_wet_fraction': float(np.median(wet)),
                                      'share_events_99pct_wet': float(np.mean(np.array(wet) >= 0.99)),
                                      'median_allwet_csi': float(np.median(allwet)),
                                      'wet_fraction_per_event': wet, 'allwet_csi_per_event': allwet}
            print(f"    {te:7s} truly wet at final step: median {np.median(wet):.3f} | "
                  f"all-wet CSI median {np.median(allwet):.4f}")
        for tr in REAL:
            for seed in (0, 1, 2):
                p = ck(tr, seed)
                if not os.path.exists(p):
                    print(f"    missing {p}")
                    continue
                m = build_model('vector', 64, 3, device)
                load_ckpt(m, p, device)
                m.eval()
                row = {'train_on': tr, 'seed': seed}
                for te in REAL:
                    r = evaluate(m, cache[te]['test'], device, max_steps=40, max_samples=19)
                    row[te] = {'csi_final': float(r['csi_final']), 'volume_ratio': float(r['volume_ratio'])}
                real['models'].append(row)
                print(f"    {tr:7s} s{seed}: on breach CSI {row['breach']['csi_final']:.4f} "
                      f"vol {row['breach']['volume_ratio']:.2f}x | on harvey CSI "
                      f"{row['harvey']['csi_final']:.4f} vol {row['harvey']['volume_ratio']:.2f}x")
    print("\n  SUMMARY (medians across seeds)")
    for te in REAL:
        line = f"    test {te:7s} all-wet CSI {real['saturation'][te]['median_allwet_csi']:.4f}"
        for tr in REAL:
            v = [r[te]['csi_final'] for r in real['models'] if r['train_on'] == tr]
            if v:
                line += f" | {tr}-trained CSI {np.median(v):.4f}"
        print(line)
    json.dump(real, open(OUT_JSON, 'w'), indent=2)
    print(f"  saved {OUT_JSON}")
    return real


def main():
    real = part_a()
    s4 = json.load(open(SEC4))
    rows, sat = s4['rows'], s4['saturation']

    st = house_style()
    print("\n  house colours found:", st)
    INK = st.get('INK', '#222222')
    MUTED = st.get('MUTED', '#8a8a8a')
    BG = next((v for k, v in st.items() if any(s in k.upper() for s in ('BG', 'PAPER', 'BACK'))), '#faf8f3')
    used = {st.get('BREACH', '').lower(), st.get('HARVEY', '').lower()}
    free = [c for c in ['#7b3294', '#1b7837', '#e08214', '#2c7fb8', '#d95f02', '#c2a5cf']
            if c.lower() not in used]
    COL = {'point': free[0], 'distributed': free[1], 'inflow': free[2],
           'breach': st.get('BREACH', '#b2182b'), 'harvey': st.get('HARVEY', '#2166ac')}

    def style(ax):
        ax.set_facecolor(BG)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=INK)

    def save(fig, name):
        for ext in ('pdf', 'png'):
            p = os.path.join(FIG, f'{name}.{ext}')
            if os.path.exists(p):
                print(f"  {p} exists - not overwritten")
                continue
            fig.savefig(p, dpi=300, bbox_inches='tight', facecolor=BG)
            print(f"  saved {p}")

    # ---- figJ ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.patch.set_facecolor(BG)
    pos = {(tr, te): gi * 4 + ti for gi, tr in enumerate(SYN) for ti, te in enumerate(SYN)}
    for key, ax in zip(['csi_final', 'volume_ratio'], axes):
        style(ax)
        for (tr, te), x in pos.items():
            vals = [float(r['cells'][te][key]) for r in rows if r['train_on'] == tr]
            ax.scatter([x] * len(vals), vals, s=14, color=COL[te], alpha=0.35, zorder=2)
            ax.scatter([x], [np.median(vals)], s=75, color=COL[te],
                       edgecolor=INK if tr == te else 'none', linewidth=1.8, zorder=3)
            if key == 'csi_final':
                ax.hlines(float(sat[te]['median_allwet_csi']), x - 0.4, x + 0.4,
                          colors=INK, linestyles=':', linewidth=1.3, zorder=1)
        if key == 'volume_ratio':
            ax.set_yscale('log')
            ax.axhspan(0.5, 2.0, color=MUTED, alpha=0.15, zorder=0)
            ax.axhline(1.0, color=MUTED, lw=0.8)
            ax.set_ylabel('Volume ratio (predicted / true)', color=INK)
        else:
            ax.set_ylim(0, 1.05)
            ax.set_ylabel('CSI (1% relative threshold)', color=INK)
        ax.set_xticks(list(pos.values()))
        ax.set_xticklabels([te for (tr, te) in pos], fontsize=7, rotation=35)
        for gi, tr in enumerate(SYN):
            ax.text(gi * 4 + 1, -0.30, f'trained on {tr}', transform=ax.get_xaxis_transform(),
                    ha='center', color=INK, fontsize=9)
    handles = ([Line2D([], [], marker='o', ls='', color=COL[m], label=f'tested on {m}') for m in SYN]
               + [Line2D([], [], marker='o', ls='', color='white', markeredgecolor=INK, label='own flood type'),
                  Line2D([], [], ls=':', color=INK, label='trivial all-wet CSI'),
                  Patch(color=MUTED, alpha=0.15, label='volume within 2x')])
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    save(fig, 'figJ_synthetic_three_mechanisms')
    plt.close(fig)

    # ---- figK ----
    tests = ['point', 'inflow', 'distributed', 'breach', 'harvey']
    satall = {**{t: sat[t] for t in SYN}, **real['saturation']}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    fig.patch.set_facecolor(BG)
    ax = axes[0]
    style(ax)
    ax.bar(range(5), [satall[t]['median_true_wet_fraction'] for t in tests],
           color=[COL[t] for t in tests])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Share of cells truly wet (final step)', color=INK)
    ax = axes[1]
    style(ax)
    for i, t in enumerate(tests):
        ax.scatter(i, satall[t]['median_allwet_csi'], marker='_', s=500, color=INK, linewidths=2.5, zorder=3)
        trainers = SYN if t in SYN else REAL
        for tr in trainers:
            if t in SYN:
                v = [float(r['cells'][t]['csi_final']) for r in rows if r['train_on'] == tr]
            else:
                v = [r[t]['csi_final'] for r in real['models'] if r['train_on'] == tr]
            if v:
                ax.scatter(i + 0.12 * (trainers.index(tr) - 1), np.median(v), s=45, color=COL[tr],
                           edgecolor=INK if tr == t else 'none', zorder=4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('CSI', color=INK)
    for a in axes:
        a.set_xticks(range(5))
        a.set_xticklabels(['point\n(synthetic)', 'inflow\n(synthetic)', 'distributed\n(synthetic)',
                           'dam-break\n(real)', 'Harvey\n(real)'], fontsize=8)
    handles = ([Line2D([], [], marker='_', ls='', color=INK, markersize=14, markeredgewidth=2.5,
                       label='trivial all-wet CSI')]
               + [Line2D([], [], marker='o', ls='', color=COL[m], label=f'trained on {m}')
                  for m in SYN + REAL])
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, 'figK_csi_saturation')
    plt.close(fig)


if __name__ == '__main__':
    main()
