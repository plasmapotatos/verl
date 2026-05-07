"""RL-induced outcome changes by reward variant (grouped bar chart, two panels).

Shows Post RL - Post SFT delta in percentage points for three outcome categories
(Correct, Incorrect, Not attempted) across four reward variants, split by
question target type (Answer targets vs Refusal targets).

Sources:
    - All values read from abstention_delta_vertical_draft.png (Overleaf assets).
    - Correct values for Refusal targets back-calculated to sum ≈ 0.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from figures.plot_utils import apply_paper_style, make_fig, top_legend, save_fig
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── colors (semantic: Correct / Incorrect / Not attempted) ─────────────────
# Okabe-Ito-inspired, colorblind-safe and semantic:
#   green = correct, vermillion = incorrect, sky-blue = abstain
COLOR_CORRECT  = "#7ab098"   # muted sage
COLOR_WRONG    = "#b56a6a"   # dusty rose
COLOR_REFUSED  = "#7a96b0"   # soft slate blue

# ── data ────────────────────────────────────────────────────────────────────
VARIANTS = ["Binary", "Ternary", "Ternary\np=0.1", "Ternary\np=0.5"]

# Answer-target panel  (correct, incorrect, not-attempted)
ANS_CORRECT      = [-1.1, -13.4, -13.1, -14.9]
ANS_INCORRECT    = [-6.3, -25.0, -25.4, -23.1]
ANS_NOT_ATTEMPT  = [ 7.5,  38.4,  38.4,  38.1]

# Refusal-target panel
REF_CORRECT      = [-0.4,  -1.1,  -1.1,  -1.2]
REF_INCORRECT    = [-3.4, -18.0, -18.0, -18.3]
REF_NOT_ATTEMPT  = [ 3.7,  19.1,  19.1,  19.5]

OUT = Path(__file__).parent.parent / "abstention_delta_vertical.png"


def _draw_panel(ax, correct, incorrect, not_attempt, title):
    n = len(VARIANTS)
    bw = 0.26          # bar width
    off = 0.3         # within-group offset between bar centers
    gap = 1          # group spacing
    xs = np.arange(n) * gap

    # three bars per group
    ax.bar(xs - off,    correct,     bw, color=COLOR_CORRECT,  label="Correct")
    ax.bar(xs,          incorrect,   bw, color=COLOR_WRONG,    label="Incorrect")
    ax.bar(xs + off,    not_attempt, bw, color=COLOR_REFUSED,  label="Not attempted")

    # zero baseline
    ax.axhline(0, color="#888888", lw=0.8, zorder=0)

    # value labels
    def _label(xpos, val):
        sign = "+" if val >= 0 else ""
        offset = 0.8 if val >= 0 else -0.8
        va = "bottom" if val >= 0 else "top"
        color = COLOR_CORRECT if xpos in [xs[i] - bw for i in range(n)] else (
                COLOR_WRONG   if xpos in list(xs)               else COLOR_REFUSED)
        ax.text(xpos, val + offset, f"{sign}{val}", ha="center", va=va,
                fontsize=9, color=color)

    for i in range(n):
        for xp, vals in [(xs[i] - off, correct), (xs[i], incorrect), (xs[i] + off, not_attempt)]:
            v = vals[i]
            sign = "+" if v >= 0 else ""
            offset = 1.0 if v >= 0 else -1.0
            va = "bottom" if v >= 0 else "top"
            ax.text(xp, v + offset, f"{sign}{v}", ha="center", va=va,
                    fontsize=15, color="black")

    ax.set_xticks(xs)
    ax.set_xticklabels(VARIANTS, fontsize=20)
    ax.set_ylim(-42, 48)
    ax.set_title(title, fontweight="bold", pad=4, fontsize=24)
    ax.set_xlim(xs[0] - 0.65, xs[-1] + 0.65)
    ax.tick_params(axis="y", labelsize=18)


def main():
    apply_paper_style(background="white")
    
    plt.rcParams.update({
        "font.size":         18,
        "axes.labelsize":    16,
        "axes.titlesize":    14,
        "xtick.labelsize":   14,
        "ytick.labelsize":   14,
        "legend.fontsize":   14,
    })

    fig, axes = make_fig(figsize=(12, 5.0), ncols=2)
    ax_ans, ax_ref = axes

    _draw_panel(ax_ans, ANS_CORRECT, ANS_INCORRECT, ANS_NOT_ATTEMPT,
                "Answer Target")
    _draw_panel(ax_ref, REF_CORRECT, REF_INCORRECT, REF_NOT_ATTEMPT,
                "Abstention Target")

    ax_ans.set_ylabel("Post RL − Post SFT (pp)", fontsize=20)
    ax_ref.set_ylabel("")

    # in-figure legend on the right panel (upper-right empty space)
    handles = [
        mpatches.Patch(color=COLOR_CORRECT, label="Correct"),
        mpatches.Patch(color=COLOR_WRONG,   label="Incorrect"),
        mpatches.Patch(color=COLOR_REFUSED, label="Not attempted"),
    ]
    ax_ref.legend(handles=handles, loc="upper left", ncol=1,
                  fontsize=15, frameon=True, framealpha=0.95,
                  borderpad=0.4, handlelength=1.4)

    save_fig(fig, OUT)


if __name__ == "__main__":
    main()
