"""
Figure: Outcome distribution shift from post-SFT to post-RL
for questions trained to answer vs. questions trained to abstain.

Usage:
    python plot_behavior_shift.py

Outputs:
    behavior_shift.pdf  (for LaTeX)
    behavior_shift.png  (for preview)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────

# (a) Questions trained to answer
answer_labels   = ["Correct", "Abstained", "Incorrect"]
answer_sft      = [42, 24, 34]
answer_rl       = [42, 31, 27]
answer_deltas   = [None, +7, -7]          # None = no annotation
answer_delta_colors = [None, "red", "green"]

# (b) Questions trained to abstain
abstain_labels  = ["Abstained", "Answered"]
abstain_sft     = [83, 17]
abstain_rl      = [92,  8]
abstain_deltas  = [+9, -9]
abstain_delta_colors = ["red", "green"]

# ── Style ─────────────────────────────────────────────────────────────────────

COLOR_SFT   = "#b0bec5"   # light blue-grey
COLOR_RL    = "#1a3a5c"   # dark navy
BAR_WIDTH   = 0.18
FONT_FAMILY = "DejaVu Sans"

plt.rcParams.update({
    "font.family":      FONT_FAMILY,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "figure.dpi":       150,
    "xtick.labelsize":  6.5,
    "ytick.labelsize":  6.5,
})

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(4.8, 2.4),
                                 gridspec_kw={"width_ratios": [3, 2]})
fig.subplots_adjust(wspace=0.4, bottom=0.28)

# ── Helper: draw delta bracket ────────────────────────────────────────────────

def draw_delta(ax, x_center, y_lo, y_hi, delta, color, bar_width=BAR_WIDTH):
    """Draw a bracket + label between two bars."""
    x_r = x_center + bar_width / 2 + 0.08   # just right of the RL bar
    tick = 0.015
    sign = "+" if delta > 0 else ""
    ax.annotate("", xy=(x_r, y_hi), xytext=(x_r, y_lo),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.2))
    ax.plot([x_r - tick, x_r + tick], [y_lo, y_lo], color=color, lw=1.2)
    ax.plot([x_r - tick, x_r + tick], [y_hi, y_hi], color=color, lw=1.2)
    ax.text(x_r + 0.04, (y_lo + y_hi) / 2,
            f"{sign}{delta}pp", color=color,
            va="center", ha="left", fontsize=9, fontweight="bold")

# ── Panel (a) ─────────────────────────────────────────────────────────────────

GROUP_SPACING = 0.65
x_a = np.arange(len(answer_labels)) * GROUP_SPACING

bars_sft_a = ax_a.bar(x_a - BAR_WIDTH/2, answer_sft, BAR_WIDTH,
                       color=COLOR_SFT, label="post-SFT")
bars_rl_a  = ax_a.bar(x_a + BAR_WIDTH/2, answer_rl,  BAR_WIDTH,
                       color=COLOR_RL,  label="post-RL")

# Value labels
for bar in bars_sft_a:
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
              f"{int(bar.get_height())}%", ha="center", va="bottom", fontsize=6)
for bar in bars_rl_a:
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
              f"{int(bar.get_height())}%", ha="center", va="bottom", fontsize=6)

ax_a.yaxis.grid(True, linestyle="--", alpha=0.6)
ax_a.xaxis.grid(False)
ax_a.set_xticks(x_a)
ax_a.set_xticklabels(answer_labels, fontsize=7)
ax_a.set_xlabel("Outcome", fontsize=7, labelpad=3)
ax_a.set_ylabel("Percentage (%)", fontsize=7)
ax_a.set_ylim(0, 65)
ax_a.set_xlim(x_a[0] - 0.45, x_a[-1] + 0.45)
ax_a.set_title(
    "(a) Trained to Answer",
    fontsize=7.5, fontweight="bold", pad=4
)
# ── Panel (b) ─────────────────────────────────────────────────────────────────

x_b = np.arange(len(abstain_labels)) * GROUP_SPACING

bars_sft_b = ax_b.bar(x_b - BAR_WIDTH/2, abstain_sft, BAR_WIDTH,
                       color=COLOR_SFT, label="post-SFT")
bars_rl_b  = ax_b.bar(x_b + BAR_WIDTH/2, abstain_rl,  BAR_WIDTH,
                       color=COLOR_RL,  label="post-RL")

for bar in bars_sft_b:
    ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
              f"{int(bar.get_height())}%", ha="center", va="bottom", fontsize=6)
for bar in bars_rl_b:
    ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
              f"{int(bar.get_height())}%", ha="center", va="bottom", fontsize=6)

ax_b.yaxis.grid(True, linestyle="--", alpha=0.6)
ax_b.xaxis.grid(False)
ax_b.set_xticks(x_b)
ax_b.set_xticklabels(abstain_labels, fontsize=7)
ax_b.set_xlabel("Outcome", fontsize=7, labelpad=3)
ax_b.set_ylabel("Percentage (%)", fontsize=7)
ax_b.set_ylim(0, 110)
ax_b.set_xlim(x_b[0] - 0.45, x_b[-1] + 0.45)
ax_b.set_title(
    "(b) Trained to Abstain",
    fontsize=7.5, fontweight="bold", pad=4
)
# ── Shared legend ─────────────────────────────────────────────────────────────

legend_handles = [
    mpatches.Patch(color=COLOR_SFT, label="post-SFT"),
    mpatches.Patch(color=COLOR_RL,  label="post-RL"),
]
fig.legend(handles=legend_handles, loc="lower center",
           ncol=2, fontsize=7, frameon=True,
           bbox_to_anchor=(0.5, 0.01))

# ── Save ──────────────────────────────────────────────────────────────────────

pdf_path = OUTPUT_DIR / "behavior_shift.pdf"
png_path = OUTPUT_DIR / "behavior_shift.png"
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, bbox_inches="tight", dpi=200)
print(f"Saved {pdf_path} and {png_path}")
plt.show()