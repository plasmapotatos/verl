"""Set diagram showing the abstention experiment scope.

Nested set rectangles:
    SimpleQA-known (excluded)  |  RichQA-unknown ⊃ SimpleQA-unknown
A dashed vertical divider runs the full vertical extent of the RichQA-unknown
box (broken only by the SimpleQA-unknown title) and splits BOTH outer
(RichQA-unknown) and inner (SimpleQA-unknown) into answer-target /
abstention-target halves (50/50). The two target labels sit outside both
boxes to make clear that the split applies to the whole region.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from figures.plot_utils import (
    apply_bar_rcparams, make_fig, save_fig,
    BAR_COLOR_SFT, BAR_COLOR_RL,
)
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent.parent / "abstention_diagram.png"

# Palette echoes refusal_50_50_behavior_shift.py: light grey-blue + dark navy.
EDGE       = BAR_COLOR_RL          # "#1a3a5c"  navy outline / accent text
TEXT_DARK  = BAR_COLOR_RL
TEXT_BODY  = "#34495e"
EXCL_FACE  = "#dde3e8"             # tint of BAR_COLOR_SFT
EXCL_EDGE  = "#6b7d88"
RICH_FACE  = "#eef3f7"             # very light blue tint
INNER_FACE = "white"


def rrect(ax, x, y, w, h, r, **kwargs):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}", **kwargs))


def main():
    apply_bar_rcparams()

    fig, ax = make_fig(figsize=(3.5, 2.5))
    ax.set_xlim(0.02, 3.40)
    ax.set_ylim(0.36, 2.36)
    ax.axis('off')

    r = 0.12

    # ── Box geometry ─────────────────────────────────────────────────────────
    # RichQA-unknown outer box.
    box_y, box_h = 0.55, 1.65
    outer_x, outer_w = 0.92, 2.45
    outer_top = box_y + box_h
    outer_cx = outer_x + outer_w / 2

    # SimpleQA-unknown inner box: vertically centered inside the outer box.
    inner_w, inner_h = 1.52, 1.05
    inner_x = outer_x + (outer_w - inner_w) / 2
    inner_y = box_y + (box_h - inner_h) / 2
    inner_cx = inner_x + inner_w / 2  # equals outer_cx
    inner_cy = inner_y + inner_h / 2

    # SimpleQA-known (excluded) box: same height as the inner box, padded
    # horizontally on both sides so it doesn't hug the figure edge or the
    # RichQA-unknown box.
    excl_x, excl_w = 0.08, 0.78
    excl_y, excl_h = inner_y, inner_h

    # ── SimpleQA-known (excluded, left) ──────────────────────────────────────
    rrect(ax, excl_x, excl_y, excl_w, excl_h, r,
          facecolor=EXCL_FACE, edgecolor=EXCL_EDGE,
          linewidth=1.0, linestyle='--')
    ax.text(excl_x + excl_w / 2, excl_y + excl_h / 2,
            'SimpleQA known\n(excluded)',
            ha='center', va='center',
            fontsize=6.6, color=TEXT_BODY)

    # ── RichQA outer box ─────────────────────────────────────────────────────
    rrect(ax, outer_x, box_y, outer_w, box_h, r,
          facecolor=RICH_FACE, edgecolor=EDGE, linewidth=1.1)
    rich_label_y = outer_top - 0.18
    ax.text(outer_cx, rich_label_y,
            'RichQA (Superset of SimpleQA)',
            ha='center', va='center',
            fontsize=7.8, fontweight='bold', color=TEXT_DARK)

    # ── SimpleQA-unknown inner box ───────────────────────────────────────────
    rrect(ax, inner_x, inner_y, inner_w, inner_h, r,
          facecolor=INNER_FACE, edgecolor=EDGE, linewidth=1.0, zorder=3)
    ax.text(inner_cx, inner_cy,
            'SimpleQA unknown',
            ha='center', va='center',
            fontsize=9.0, fontweight='bold',
            color=TEXT_DARK, zorder=5)

    # ── Dashed divider — broken around both labels for readability ──────────
    rich_gap_half = 0.11
    inner_gap_half = 0.13
    ax.plot([inner_cx, inner_cx], [box_y, inner_cy - inner_gap_half],
            color=EDGE, linewidth=0.9, linestyle='--', zorder=4)
    ax.plot([inner_cx, inner_cx], [inner_cy + inner_gap_half, rich_label_y - rich_gap_half],
            color=EDGE, linewidth=0.9, linestyle='--', zorder=4)
    ax.plot([inner_cx, inner_cx], [rich_label_y + rich_gap_half, outer_top],
            color=EDGE, linewidth=0.9, linestyle='--', zorder=4)

    # ── Target labels: inside the RichQA box, one per half ──────────────────
    lower_band_y = box_y + 0.11
    left_half_cx = (outer_x + inner_cx) / 2
    right_half_cx = (inner_cx + outer_x + outer_w) / 2
    ax.text(left_half_cx, lower_band_y,
            'Answer Target',
            ha='center', va='bottom',
            fontsize=7.4, fontweight='bold', color=TEXT_DARK)
    ax.text(right_half_cx, lower_band_y,
            'Abstention Target',
            ha='center', va='bottom',
            fontsize=7.1, fontweight='bold', color=TEXT_DARK)

    save_fig(fig, OUT)


if __name__ == "__main__":
    main()
