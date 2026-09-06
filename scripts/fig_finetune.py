"""
fig_finetune.py
===============

FIGURE 5 (fine-tuning): how performance changes when you give a
transferred model a few examples of the new flood type.

Shows the Harvey->Breach direction (the clean one), comparing:
  - before fine-tuning (zero target samples)
  - after 5 target samples
  - after 10 target samples

Reads the real result JSON files. Run AFTER the fine-tuning experiments.

Output: results/figures/fig5_finetune.png and .pdf

Run:
    python3 scripts/fig_finetune.py
"""

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_DIR = 'results/figures'
RESULTS_DIR = 'results'
os.makedirs(FIG_DIR, exist_ok=True)


def collect_finetune(direction='harveytobreach'):
    """
    Collect before/after CSI for the fine-tuning runs in one direction.
    Returns dict: {0: [csi...], 5: [csi...], 10: [csi...]}
    where 0 = before fine-tuning.
    """
    data = {0: [], 5: [], 10: []}

    for path in glob.glob(os.path.join(RESULTS_DIR, f'result_finetune_{direction}_*.json')):
        with open(path) as f:
            r = json.load(f)
        n = r['ft_samples']
        before = r['before_ft_target']['csi_final']
        after = r['after_ft_target']['csi_final']
        # 'before' is the same regardless of n, collect once per file
        data[0].append(before)
        data[n].append(after)

    return data


def main():
    data = collect_finetune('harveytobreach')

    sample_counts = [0, 5, 10]
    means = [np.mean(data[n]) if data[n] else 0 for n in sample_counts]
    stds = [np.std(data[n]) if data[n] else 0 for n in sample_counts]

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(sample_counts))
    colors = ['#B0BEC5', '#66BB6A', '#43A047']

    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                  edgecolor='#2C3E50', lw=1.5, alpha=0.9, width=0.6)

    # Connect with a trend line
    ax.plot(x, means, 'o-', color='#1565C0', lw=2, markersize=8, zorder=5)

    # Value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.008, f'{m:.3f}', ha='center', fontsize=12,
                fontweight='bold', color='#2C3E50')

    ax.set_xticks(x)
    ax.set_xticklabels(['0\n(no fine-tuning)', '5 samples', '10 samples'],
                       fontsize=11)
    ax.set_xlabel('Number of target-type examples used for fine-tuning',
                  fontsize=12)
    ax.set_ylabel('CSI on the new flood type (higher = better)', fontsize=12)
    ax.set_title('A Few Examples of the New Flood Type Help\n'
                 '(Rainfall-trained model adapting to dam-break floods)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(means) * 1.4 if max(means) > 0 else 0.3)

    fig.text(0.5, -0.02,
             'Even five examples of the new flood type produce a consistent '
             'improvement — a cheap, practical fix.',
             ha='center', fontsize=10, style='italic', color='#2C3E50')

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f'fig5_finetune.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close()

    print("\nFine-tuning (Harvey->Breach) mean CSI:")
    for n in sample_counts:
        print(f"  {n} samples: {means[sample_counts.index(n)]:.4f}")


if __name__ == '__main__':
    main()
