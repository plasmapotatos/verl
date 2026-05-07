"""Outcome distribution shift from post-SFT (step 0) to post-RL (step 680).

Run: richqa_grpo_refusal_50_50 (validation split)

Sources:
    - outputs/rl/richqa_grpo_refusal_50_50/binary/global_step_{0,680}/generations/
        binary_global_step_{step}__on_val_{answer,refusal}_eval.json

Counts (out of 275 each):
    - answer split (trained to answer):
        step 0  : correct=95,  not_attempted=95,  incorrect=85
        step 680: correct=92,  not_attempted=106, incorrect=77
    - refusal split (trained to abstain):
        step 0  : not_attempted=250, attempted=25
        step 680: not_attempted=260, attempted=15
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


OUT = Path(__file__).parent.parent / "refusal_50_50_behavior_shift.png"


def _pct(n, denom=275):
    return 100.0 * (n / denom)


def _fmt(v: float) -> str:
    return f"{v:.1f}"


def main():
    apply_bar_rcparams()
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    bar_w = BAR_WIDTH * 0.10
    group_spacing = BAR_WIDTH * 0.18 * 1.6  # same absolute spacing as before

    # ── Data (percent) ──────────────────────────────────────────────────────
    answer_labels = ["Correct", "Incorrect", "Abstained"]
    answer_sft = [37.7, 37.3, 25.0]
    answer_rl = [36.6, 31.0, 32.5]

    abstain_labels = ["Correct", "Incorrect", "Abstained"]
    abstain_sft = [0.0, 12.4, 87.6]
    abstain_rl = [0.0, 8.6, 91.4]

    # ── Layout ─────────────────────────────────────────────────────────────
    fig, (ax_a, ax_b) = make_fig(
        figsize=(7.2, 3.2),
        ncols=2,
        gridspec_kw={"width_ratios": [1, 1]},
    )
    fig.subplots_adjust(wspace=0.45)

    # Put horizontal grid/spine lines behind the bars.
    for ax in (ax_a, ax_b):
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", zorder=0)
        for side in ("bottom", "left"):
            ax.spines[side].set_zorder(0)

    # ── Panel (a): trained to answer ───────────────────────────────────────
    x_a = np.arange(len(answer_labels)) * group_spacing
    bars_sft_a = ax_a.bar(
        x_a - bar_w / 2,
        answer_sft,
        bar_w,
        color=BAR_COLOR_SFT,
        label="post-SFT",
        zorder=3,
    )
    bars_rl_a = ax_a.bar(
        x_a + bar_w / 2,
        answer_rl,
        bar_w,
        color=BAR_COLOR_RL,
        label="post-RL",
        zorder=3,
    )

    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(answer_labels)
    ax_a.set_ylabel("Percentage (%)")
    ax_a.set_ylim(0, 50)
    ax_a.set_title("(a) Answer Target")

    for bar in list(bars_sft_a) + list(bars_rl_a):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            _fmt(bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=12,
            zorder=4,
        )

    # ── Panel (b): trained to abstain ──────────────────────────────────────
    x_b = np.arange(len(abstain_labels)) * group_spacing
    bars_sft_b = ax_b.bar(
        x_b - bar_w / 2,
        abstain_sft,
        bar_w,
        color=BAR_COLOR_SFT,
        label="post-SFT",
        zorder=3,
    )
    bars_rl_b = ax_b.bar(
        x_b + bar_w / 2,
        abstain_rl,
        bar_w,
        color=BAR_COLOR_RL,
        label="post-RL",
        zorder=3,
    )

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(abstain_labels)
    ax_b.set_ylabel("Percentage (%)")
    ax_b.set_ylim(0, 105)
    ax_b.set_title("(b) Abstention Target")

    for bar in list(bars_sft_b) + list(bars_rl_b):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            _fmt(bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=12,
            zorder=4,
        )

    ax_b.legend(loc="upper left", ncol=1, fontsize=14, framealpha=0.9)
    save_fig(fig, OUT)


if __name__ == "__main__":
    main()
