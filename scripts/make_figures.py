"""
make_figures.py
===============

Produce the three key figures for the paper from the 12 result JSON files.

Usage:
    python3 scripts/make_figures.py

Outputs go to results/figures/
"""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no display needed on a server
import matplotlib.pyplot as plt

RESULTS_DIR = 'results'
FIG_DIR = 'results/figures'
os.makedirs(FIG_DIR, exist_ok=True)


def load_results():
    """Load all 12 result JSON files."""
    results = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, 'result_*.json'))):
        with open(path) as f:
            r = json.load(f)
        results[r['tag']] = r
    print(f"Loaded {len(results)} results: {list(results.keys())}")
    return results


def figure1_transfer_gap_bars(results):
    """
    Figure 1: Home vs Away CSI for Harvey-trained models (the clean direction).
    
    This is the paper's headline figure. It shows the ~70% skill drop
    when a Harvey-trained model is tested on breach data.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    groups = ['vector', 'scalar']
    x = np.arange(len(groups))
    width = 0.35

    home_means = []
    away_means = []
    home_stds = []
    away_stds = []

    for variant in groups:
        homes = []
        aways = []
        for seed in [0, 1, 2]:
            tag = f"harvey_{variant}_s{seed}"
            if tag in results:
                homes.append(results[tag]['home_scores']['csi_final'])
                aways.append(results[tag]['away_scores']['csi_final'])
        home_means.append(np.mean(homes))
        away_means.append(np.mean(aways))
        home_stds.append(np.std(homes))
        away_stds.append(np.std(aways))

    bars1 = ax.bar(x - width/2, home_means, width, yerr=home_stds,
                   label='Home (Harvey)', color='#2196F3', capsize=5, alpha=0.85)
    bars2 = ax.bar(x + width/2, away_means, width, yerr=away_stds,
                   label='Away (Breach)', color='#FF9800', capsize=5, alpha=0.85)

    # Add retained % labels
    for i in range(len(groups)):
        if home_means[i] > 0:
            pct = 100 * away_means[i] / home_means[i]
            ax.annotate(f'{pct:.0f}% retained',
                       xy=(x[i] + width/2, away_means[i] + away_stds[i] + 0.02),
                       ha='center', fontsize=10, color='#E65100', fontweight='bold')

    ax.set_ylabel('CSI (flood map accuracy)', fontsize=12)
    ax.set_title('Cross-Type Transfer: Harvey-Trained Models on Breach Data',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Vector\n(keeps direction)', 'Scalar\n(speed only)'],
                       fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 0.7)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig1_transfer_gap.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close()


def figure2_csi_vs_mass(results):
    """
    Figure 2: CSI vs Mass Error for ALL 12 runs.
    
    This is Finding 2's visual proof. Home runs cluster near the origin.
    Breach-to-Harvey runs sit at high CSI + extreme mass error, showing
    how CSI can hide physically nonsensical predictions.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    markers = {'vector': 'o', 'scalar': 's'}
    
    for tag, r in results.items():
        train_on = r['train_on']
        variant = r['variant']
        
        # Home point
        h_csi = r['home_scores']['csi_final']
        h_mass = r['home_scores']['mass_error']
        
        # Away point
        a_csi = r['away_scores']['csi_final']
        a_mass = r['away_scores']['mass_error']
        
        color_home = '#2196F3' if train_on == 'harvey' else '#4CAF50'
        color_away = '#FF9800' if train_on == 'harvey' else '#F44336'
        
        m = markers[variant]
        
        ax.scatter(h_csi, h_mass, marker=m, c=color_home, s=80, alpha=0.7,
                  edgecolors='black', linewidth=0.5)
        ax.scatter(a_csi, a_mass, marker=m, c=color_away, s=80, alpha=0.7,
                  edgecolors='black', linewidth=0.5)

    # Legend entries (manual)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
               markersize=10, label='Harvey-trained, home'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9800',
               markersize=10, label='Harvey-trained, away (breach)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50',
               markersize=10, label='Breach-trained, home'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336',
               markersize=10, label='Breach-trained, away (Harvey)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markersize=8, label='Circle = vector'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='grey',
               markersize=8, label='Square = scalar'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left')

    ax.set_xlabel('CSI (higher = better spatial accuracy)', fontsize=12)
    ax.set_ylabel('Mass Error (0 = perfect water balance)', fontsize=12)
    ax.set_title('CSI vs Mass Error: How CSI Can Hide Transfer Failure',
                 fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='grey', linestyle='--', alpha=0.5)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2_csi_vs_mass.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close()


def figure3_breach_to_harvey_trap(results):
    """
    Figure 3: The CSI trap — Breach→Harvey looks good on CSI but catastrophic
    on mass error. Side-by-side bars.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    groups = ['vector', 'scalar']
    x = np.arange(len(groups))
    width = 0.35

    # CSI panel
    home_csi = []
    away_csi = []
    for variant in groups:
        h = [results[f'breach_{variant}_s{s}']['home_scores']['csi_final'] for s in range(3)]
        a = [results[f'breach_{variant}_s{s}']['away_scores']['csi_final'] for s in range(3)]
        home_csi.append((np.mean(h), np.std(h)))
        away_csi.append((np.mean(a), np.std(a)))

    ax1.bar(x - width/2, [h[0] for h in home_csi], width,
            yerr=[h[1] for h in home_csi],
            label='Home (Breach)', color='#4CAF50', capsize=5, alpha=0.85)
    ax1.bar(x + width/2, [a[0] for a in away_csi], width,
            yerr=[a[1] for a in away_csi],
            label='Away (Harvey)', color='#F44336', capsize=5, alpha=0.85)
    ax1.set_ylabel('CSI', fontsize=12)
    ax1.set_title('CSI looks like improvement', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Vector', 'Scalar'], fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 0.7)
    ax1.grid(axis='y', alpha=0.3)

    # Mass error panel
    home_mass = []
    away_mass = []
    for variant in groups:
        h = [results[f'breach_{variant}_s{s}']['home_scores']['mass_error'] for s in range(3)]
        a = [results[f'breach_{variant}_s{s}']['away_scores']['mass_error'] for s in range(3)]
        home_mass.append((np.mean(h), np.std(h)))
        away_mass.append((np.mean(a), np.std(a)))

    ax2.bar(x - width/2, [h[0] for h in home_mass], width,
            yerr=[h[1] for h in home_mass],
            label='Home (Breach)', color='#4CAF50', capsize=5, alpha=0.85)
    ax2.bar(x + width/2, [a[0] for a in away_mass], width,
            yerr=[a[1] for a in away_mass],
            label='Away (Harvey)', color='#F44336', capsize=5, alpha=0.85)
    ax2.set_ylabel('Mass Error (lower = better)', fontsize=12)
    ax2.set_title('Mass error reveals the truth', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Vector', 'Scalar'], fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('The CSI Trap: Breach-Trained Models on Harvey Data',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig3_csi_trap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {path}")
    plt.close()


def print_summary_table(results):
    """Print the full results table for the report."""
    print("\n" + "=" * 90)
    print("  FULL RESULTS TABLE")
    print("=" * 90)
    print(f"  {'Tag':<25s} {'Home CSI':>10s} {'Away CSI':>10s} {'Retained':>10s} "
          f"{'Home Mass':>10s} {'Away Mass':>10s}")
    print("-" * 90)

    for tag in sorted(results.keys()):
        r = results[tag]
        h_csi = r['home_scores']['csi_final']
        a_csi = r['away_scores']['csi_final']
        h_mass = r['home_scores']['mass_error']
        a_mass = r['away_scores']['mass_error']
        if h_csi > 0:
            retained = f"{100 * a_csi / h_csi:.1f}%"
        else:
            retained = "n/a"
        print(f"  {tag:<25s} {h_csi:>10.4f} {a_csi:>10.4f} {retained:>10s} "
              f"{h_mass:>10.1f} {a_mass:>10.1f}")


if __name__ == '__main__':
    print("Making figures ...\n")
    results = load_results()
    print_summary_table(results)
    print()
    figure1_transfer_gap_bars(results)
    figure2_csi_vs_mass(results)
    figure3_breach_to_harvey_trap(results)
    print(f"\nAll figures saved to {FIG_DIR}/")