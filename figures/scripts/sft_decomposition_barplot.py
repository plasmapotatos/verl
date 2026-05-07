"""Decomposition SFT accuracy: unfiltered vs clean-mixture variants.

Sources:
    - All values supplied directly by the user (post-SFT accuracy %).
    - RichQA baseline reference line at 43.3 also user-supplied.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from figures.plot_utils import (
    apply_paper_style,
    make_fig,
    save_fig,
    BAR_PALETTE,
    COLOR_BASELINE,
)

LABELS = ["Unfiltered", "Clean p=25%", "Clean p=50%", "Clean p=100%"]
VALUES = [56.5, 28.7, 28.0, 27.4]
BASELINE = 43.3

# Highlight the unfiltered bar with the darkest palette color; clean variants share a muted tone.
COLORS = [BAR_PALETTE[2], BAR_PALETTE[0], BAR_PALETTE[0], BAR_PALETTE[0]]

OUT = Path(__file__).parent.parent / "sft_decomposition_barplot.pdf"


def main():
    apply_paper_style()
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size":         14,
        "axes.labelsize":    14,
        "axes.titlesize":    14,
        "xtick.labelsize":   14,
        "ytick.labelsize":   14,
        "legend.fontsize":   14,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.facecolor": "white",
        "legend.facecolor": "white",
    })
    
    fig, ax = make_fig(figsize=(6, 3.4))

    xs = list(range(len(LABELS)))
    bars = ax.bar(xs, VALUES, color=COLORS, width=0.55,
                  edgecolor="white", linewidth=0.4)

    for bar, v in zip(bars, VALUES):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{v:.1f}", ha="center", va="bottom", fontsize=14)

    ax.axhline(BASELINE, linestyle="--", color=COLOR_BASELINE, linewidth=1.2,
               label="RichQA SFT (Baseline)")

    ax.set_xticks(xs)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0, 65)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    save_fig(fig, OUT)


if __name__ == "__main__":
    main()
