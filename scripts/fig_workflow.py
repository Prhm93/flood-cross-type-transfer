"""
fig_workflow.py
===============

FIGURE 2 (workflow diagram): the whole method as a clean flowchart.

Shows: two datasets -> unified format -> model training -> hour-by-hour
rollout -> scoring with two metrics -> the transfer comparison.

This is the kind of pipeline diagram every methods paper has. It lets a
reader follow the whole approach at a glance.

Output: results/figures/fig2_workflow.png and .pdf

Run:
    python3 scripts/fig_workflow.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = 'results/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Professional muted palette
C_DATA = '#E8F0F8'
C_DATA_EDGE = '#4A90D9'
C_PROCESS = '#FFF3E0'
C_PROCESS_EDGE = '#F5A623'
C_EVAL = '#E8F5E9'
C_EVAL_EDGE = '#43A047'
C_RESULT = '#FCE4EC'
C_RESULT_EDGE = '#D81B60'
C_TEXT = '#2C3E50'
C_ARROW = '#546E7A'


def box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=10, bold=False):
    """Draw a rounded box with centred text."""
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.02",
                       facecolor=facecolor, edgecolor=edgecolor, lw=2, zorder=3)
    ax.add_patch(b)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=C_TEXT, fontweight=weight, zorder=4, wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, offset=0.0):
    """Draw an arrow between two points, optional label."""
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                        mutation_scale=18, color=C_ARROW, lw=2, zorder=2)
    ax.add_patch(a)
    if text:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + offset, my, text, ha='center', va='center', fontsize=8.5,
                color=C_ARROW, style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.8))


def main():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # --- Row 1: the two datasets ---
    box(ax, 2.2, 9, 3.2, 1.1,
        'Dam-Break Floods\n(SWE-GNN data)\n60 train / 19 test',
        C_DATA, C_DATA_EDGE, fontsize=9.5, bold=True)
    box(ax, 7.8, 9, 3.2, 1.1,
        'Rainfall Floods\n(Hurricane Harvey)\n1063 train / 228 test',
        C_DATA, C_DATA_EDGE, fontsize=9.5, bold=True)

    # --- Row 2: unified format ---
    box(ax, 5, 7.3, 5.5, 1.0,
        'Unified Graph Format\ndepth + flow-direction + rainfall + terrain, same scale',
        C_DATA, C_DATA_EDGE, fontsize=9.5)

    arrow(ax, 2.2, 8.45, 4.0, 7.8)
    arrow(ax, 7.8, 8.45, 6.0, 7.8)

    # --- Row 3: model ---
    box(ax, 5, 5.6, 4.5, 1.0,
        'Graph Model Learns\n"given now, predict the next hour"',
        C_PROCESS, C_PROCESS_EDGE, fontsize=9.5, bold=True)
    arrow(ax, 5, 6.8, 5, 6.1, 'train on ONE type')

    # --- Row 4: rollout ---
    box(ax, 5, 3.9, 5.5, 1.0,
        'Hour-by-Hour Prediction (Rollout)\nfeed each prediction back in, 40 hours ahead',
        C_PROCESS, C_PROCESS_EDGE, fontsize=9.5)
    arrow(ax, 5, 5.1, 5, 4.4)

    # --- Row 5: two metrics ---
    box(ax, 2.5, 2.2, 3.6, 1.0,
        'Flood-Map Score (CSI)\nwhich cells are wet?',
        C_EVAL, C_EVAL_EDGE, fontsize=9.5)
    box(ax, 7.5, 2.2, 3.6, 1.0,
        'Water-Balance Check\nis total water sensible?',
        C_EVAL, C_EVAL_EDGE, fontsize=9.5)
    arrow(ax, 4.0, 3.4, 2.9, 2.7)
    arrow(ax, 6.0, 3.4, 7.1, 2.7)

    # --- Row 6: the comparison ---
    box(ax, 5, 0.6, 6.5, 0.9,
        'TRANSFER COMPARISON:  test on SAME type  vs  test on DIFFERENT type',
        C_RESULT, C_RESULT_EDGE, fontsize=10, bold=True)
    arrow(ax, 2.5, 1.7, 4.2, 1.05)
    arrow(ax, 7.5, 1.7, 5.8, 1.05)

    # Title
    ax.text(5, 9.9, 'Method Overview', ha='center', fontsize=16,
            fontweight='bold', color=C_TEXT)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f'fig2_workflow.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close()


if __name__ == '__main__':
    main()
