"""Outcome distribution shift (post-SFT → post-RL) for Refuse vs Refuse+Q+A paraphrase.

Four panels in a single row:
    (a) Refuse — Answer Target
    (b) Refuse — Abstention Target
    (c) Refuse + Q+A para. — Answer Target
    (d) Refuse + Q+A para. — Abstention Target
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from figures.plot_utils import (
    apply_bar_rcparams,
    make_fig,
    save_fig,
    top_legend,
    BAR_COLOR_SFT,
    BAR_COLOR_RL,
    BAR_WIDTH,
)


OUT = Path(__file__).parent.parent / "abstention_paraphrase_unstacked.png"


def _fmt(v: float) -> str:
    return f"{v:.1f}"


def _draw_panel(ax, labels, sft, rl, *, title, ylim, bar_w, group_spacing,
                ylabel="Percentage (%)"):
    x = np.arange(len(labels)) * group_spacing
    bars_sft = ax.bar(
        x - bar_w / 2, sft, bar_w,
        color=BAR_COLOR_SFT, label="post-SFT", zorder=3,
    )
    bars_rl = ax.bar(
        x + bar_w / 2, rl, bar_w,
        color=BAR_COLOR_RL, label="post-RL", zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(title)

    pad = (ylim[1] - ylim[0]) * 0.02
    for bar in list(bars_sft) + list(bars_rl):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + pad,
            _fmt(bar.get_height()),
            ha="center", va="bottom", fontsize=12, zorder=4,
        )


def main():
    apply_bar_rcparams()
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    })

    bar_w = BAR_WIDTH * 0.10
    group_spacing = BAR_WIDTH * 0.18 * 1.6

    # ── Data (percent) ─────────────────────────────────────────────────────
    labels = ["Correct", "Incorrect", "Abstained"]

    # Refuse
    refuse_answer_sft  = [37.7, 37.3, 25.0]
    refuse_answer_rl   = [36.6, 31.0, 32.5]
    refuse_abstain_sft = [0.4, 12.0, 87.6]
    refuse_abstain_rl  = [0.0,  8.6, 91.4]

    # Refuse + Q+A paraphrase
    para_answer_sft  = [50.8, 36.9, 12.3]
    para_answer_rl   = [50.0, 28.0, 22.0]
    para_abstain_sft = [0.4,  6.7, 92.9]
    para_abstain_rl  = [0.4,  2.6, 97.0]

    # ── Layout: single row of 4 panels ─────────────────────────────────────
    fig, axes = make_fig(
        figsize=(14.0, 3.4),
        ncols=4,
        gridspec_kw={"width_ratios": [1, 1, 1, 1]},
    )
    fig.subplots_adjust(wspace=0.42)

    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", zorder=0)
        for side in ("bottom", "left"):
            ax.spines[side].set_zorder(0)

    _draw_panel(axes[0], labels, refuse_answer_sft, refuse_answer_rl,
                title="(a) Answer Target (Base)",
                ylim=(0, 60), bar_w=bar_w, group_spacing=group_spacing)
    _draw_panel(axes[1], labels, para_answer_sft, para_answer_rl,
                title="(b) Answer Target (Q+A)",
                ylim=(0, 60), bar_w=bar_w, group_spacing=group_spacing)
    _draw_panel(axes[2], labels, refuse_abstain_sft, refuse_abstain_rl,
                title="(c) Abstention Target (Base)",
                ylim=(0, 105), bar_w=bar_w, group_spacing=group_spacing)
    _draw_panel(axes[3], labels, para_abstain_sft, para_abstain_rl,
                title="(d) Abstention Target (Q+A)",
                ylim=(0, 105), bar_w=bar_w, group_spacing=group_spacing)

    top_legend(fig, ncol=2, y=0.88, fontsize=14)
    save_fig(fig, OUT)


if __name__ == "__main__":
    main()
