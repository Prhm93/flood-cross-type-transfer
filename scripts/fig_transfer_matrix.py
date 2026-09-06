"""
fig_transfer_matrix.py
======================

FIGURE 3 (transfer matrix): a clean 2x2 grid showing performance when
trained on each type and tested on each type.

The diagonal (train and test on the same type) should be strong.
The off-diagonal (train on one, test on the other) shows the failure.
This is the single clearest picture of the whole finding.

Reads the real result JSON files. Run AFTER the experiments.

Output: results/figures/fig3_transfer_matrix.png and .pdf

Run:
    python3 scripts/fig_transfer_matrix.py
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


def load_single_type_results():
    """
    Build the 2x2 transfer matrix from the single-type training results.

    Matrix[train][test] = mean CSI across seeds and variants.
    """
    # Collect CSI values
    # For breach-trained: home = breach, away = harvey
    # For harvey-trained: home = harvey, away = breach
    breach_home, breach_away = [], []
    harvey_home, harvey_away = [], []

    for path in glob.glob(os.path.join(RESULTS_DIR, 'result_breach_*.json')):
        if 'mixed' in path or 'finetune' in path:
            continue
        with open(path) as f:
            r = json.load(f)
        breach_home.append(r['home_scores']['csi_final'])
        breach_away.append(r['away_scores']['csi_final'])

    for path in glob.glob(os.path.join(RESULTS_DIR, 'result_harvey_*.json')):
        if 'mixed' in path or 'finetune' in path:
            continue
        with open(path) as f:
            r = json.load(f)
        harvey_home.append(r['home_scores']['csi_final'])
        harvey_away.append(r['away_scores']['csi_final'])

    # Matrix rows = trained on, cols = tested on
    # Order: [breach, harvey]
    matrix = np.array([
        [np.mean(breach_home), np.mean(breach_away)],   # trained breach
        [np.mean(harvey_away), np.mean(harvey_home)],   # trained harvey
    ])
    return matrix


def main():
    matrix = load_single_type_results()

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # Heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=0.6, aspect='auto')

    labels = ['Dam-Break', 'Rainfall']

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    ax.set_xlabel('TESTED on', fontsize=13, fontweight='bold')
    ax.set_ylabel('TRAINED on', fontsize=13, fontweight='bold')

    # Annotate each cell
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            is_diag = (i == j)
            label = 'SAME type' if is_diag else 'DIFFERENT type'
            txtcolor = 'black'
            ax.text(j, i - 0.08, f'{val:.3f}', ha='center', va='center',
                    fontsize=20, fontweight='bold', color=txtcolor)
            ax.text(j, i + 0.15, label, ha='center', va='center',
                    fontsize=10, style='italic', color=txtcolor)

    # Draw a box around the diagonal to highlight "home" performance
    for k in range(2):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                                   edgecolor='#1565C0', lw=3, zorder=5))

    ax.set_title('Transfer Matrix: Flood-Map Accuracy (CSI)\n'
                 'Diagonal = tested on the same type it was trained on',
                 fontsize=13, fontweight='bold', pad=15)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('CSI (higher = better)', fontsize=11)

    # Caption
    fig.text(0.5, -0.03,
             'Performance drops sharply off the diagonal — models tested on a '
             'different flood type lose most of their skill.',
             ha='center', fontsize=10, style='italic', color='#2C3E50')

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f'fig3_transfer_matrix.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close()

    # Print the matrix for the report
    print("\nTransfer matrix (rows=trained, cols=tested):")
    print(f"              Test Breach   Test Rainfall")
    print(f"Train Breach     {matrix[0,0]:.3f}         {matrix[0,1]:.3f}")
    print(f"Train Rainfall   {matrix[1,0]:.3f}         {matrix[1,1]:.3f}")


if __name__ == '__main__':
    main()
