"""
fig_concept_flood_types.py
==========================

FIGURE 1 (concept diagram): the two flood types side by side.

This is NOT a bar chart. It is a schematic illustration showing the
physical difference between the two flood types — the core idea of the
whole paper — so a reader understands it in one glance.

Left panel:  dam-break flood — water enters from ONE point, spreads out
Right panel: rainfall flood  — water arrives EVERYWHERE from above

Output: results/figures/fig1_concept.png and .pdf (vector, for the paper)

Run:
    python3 scripts/fig_concept_flood_types.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

FIG_DIR = 'results/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Colour palette — muted, professional
C_WATER = '#4A90D9'
C_WATER_LIGHT = '#A8D0F0'
C_GROUND = '#D4C5A9'
C_GROUND_DARK = '#B8A582'
C_RAIN = '#5B9BD5'
C_ARROW = '#2C3E50'
C_TEXT = '#2C3E50'


def draw_terrain(ax, x0, width, base_y=0.15, seed=0):
    """Draw a gently undulating ground profile."""
    rng = np.random.RandomState(seed)
    xs = np.linspace(x0, x0 + width, 100)
    ys = base_y + 0.02 * np.sin((xs - x0) * 8) + 0.015 * rng.randn(100).cumsum() / 10
    ys = base_y + 0.03 * np.sin((xs - x0) * 6)
    ax.fill_between(xs, 0, ys, color=C_GROUND, zorder=2)
    ax.plot(xs, ys, color=C_GROUND_DARK, lw=1.5, zorder=3)
    return xs, ys


def panel_dam_break(ax):
    """Left panel: water enters from one point and spreads."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'Dam-Break Flood', ha='center', fontsize=15,
            fontweight='bold', color=C_TEXT)
    ax.text(0.5, 0.88, 'Water enters from ONE point', ha='center',
            fontsize=11, color=C_TEXT, style='italic')

    # Ground
    draw_terrain(ax, 0.05, 0.9, base_y=0.15)

    # The breach (a gap in a wall on the left)
    wall_x = 0.12
    ax.add_patch(Rectangle((wall_x - 0.02, 0.15), 0.04, 0.45,
                           facecolor='#8B7355', edgecolor='#5C4A33', lw=1.5, zorder=4))
    # gap in the wall
    ax.add_patch(Rectangle((wall_x - 0.02, 0.30), 0.04, 0.12,
                           facecolor='white', edgecolor='none', zorder=5))

    # Water spreading from the breach — concentric wavefronts
    breach_point = (wall_x + 0.02, 0.36)
    for i, r in enumerate([0.12, 0.22, 0.32, 0.42]):
        alpha = 0.5 - i * 0.09
        wedge = plt.matplotlib.patches.Wedge(
            breach_point, r, -55, 55, width=0.06,
            facecolor=C_WATER, alpha=alpha, zorder=6, edgecolor='none')
        ax.add_patch(wedge)

    # Water body near breach
    xs = np.linspace(wall_x + 0.02, 0.55, 50)
    water_top = 0.36 - 0.25 * (xs - breach_point[0])
    water_top = np.clip(water_top, 0.18, 0.42)
    ax.fill_between(xs, 0.15, water_top, color=C_WATER, alpha=0.7, zorder=5)

    # Big arrow showing flow direction from the point
    arrow = FancyArrowPatch(breach_point, (0.6, 0.30),
                            arrowstyle='-|>', mutation_scale=25,
                            color=C_ARROW, lw=2.5, zorder=10)
    ax.add_patch(arrow)
    ax.text(0.4, 0.52, 'spreads\noutward', ha='center', fontsize=10,
            color=C_ARROW, fontweight='bold')

    # Label the source
    ax.annotate('the breach\n(one source)', xy=breach_point, xytext=(0.30, 0.72),
                fontsize=10, color=C_TEXT, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_TEXT, lw=1.2))


def panel_rainfall(ax):
    """Right panel: water arrives everywhere from above."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'Rainfall Flood', ha='center', fontsize=15,
            fontweight='bold', color=C_TEXT)
    ax.text(0.5, 0.88, 'Water arrives EVERYWHERE at once', ha='center',
            fontsize=11, color=C_TEXT, style='italic')

    # Ground
    draw_terrain(ax, 0.05, 0.9, base_y=0.15, seed=3)

    # A shallow layer of water everywhere on top of the ground
    xs = np.linspace(0.05, 0.95, 100)
    ground = 0.15 + 0.03 * np.sin((xs - 0.05) * 6)
    ax.fill_between(xs, ground, ground + 0.06, color=C_WATER, alpha=0.6, zorder=5)

    # Rain — many arrows falling from the top, evenly spread
    rng = np.random.RandomState(1)
    for x in np.linspace(0.12, 0.88, 12):
        for y_off in [0, 0.12, 0.24]:
            y_top = 0.80 - y_off
            y_bot = y_top - 0.07
            ax.add_patch(FancyArrowPatch((x, y_top), (x, y_bot),
                         arrowstyle='-|>', mutation_scale=10,
                         color=C_RAIN, lw=1.5, alpha=0.75, zorder=8))

    # A cloud band across the top
    for cx in np.linspace(0.15, 0.85, 6):
        ax.add_patch(Circle((cx, 0.84), 0.06, facecolor='#B0BEC5',
                            edgecolor='none', alpha=0.6, zorder=7))

    ax.text(0.5, 0.55, 'rises evenly\neverywhere', ha='center', fontsize=10,
            color=C_ARROW, fontweight='bold')


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    panel_dam_break(ax1)
    panel_rainfall(ax2)

    # A divider and central message
    fig.text(0.5, 0.5, 'VS', ha='center', va='center', fontsize=20,
             fontweight='bold', color='#95A5A6',
             path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    fig.suptitle('Two Fundamentally Different Flood Types',
                 fontsize=17, fontweight='bold', color=C_TEXT, y=1.02)

    fig.text(0.5, -0.02,
             'A model that learns one type has no guarantee it understands the other. '
             'That is the question this paper tests.',
             ha='center', fontsize=11, style='italic', color=C_TEXT)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f'fig1_concept.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close()


if __name__ == '__main__':
    main()
