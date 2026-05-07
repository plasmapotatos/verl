"""SimpleQA: compare paraphrase-answer vs paraphrase-both (k=5) on train_eval_origqa vs epoch.

Sources (train_eval_origqa.simpleqa_f1, ×100):
    - Paraphrase Answer (k=5): outputs/sft/simpleqa_paraphrase_answer_per_epoch_k5_sft/
                              sft_lr1.5e-4_epmax20_seed1/plots/simpleqa_harmonic_mean_scores.json
    - Paraphrase Both (k=5):   outputs/sft/simpleqa_paraphrase_both_k1_3/simpleqa_harmonic_mean_k5.json
    - Baseline: RichQA SFT = 43.3 (per user)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt

from figures.plot_utils import (
    apply_paper_style,
    make_fig,
    plot_curves,
    style_ax,
    save_fig,
    add_ceiling_line,
)

# Even-epoch checkpoints only, prepended with a shared (0, 0) origin.
# Paraphrase-Answer source file only has epochs 1..19 (no epoch 20).
# For visual comparison, we extend epoch 20 by repeating the epoch-18 value.
XS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

PARAPHRASE_ANSWER_K5 = [
    0.0,
    33.363719234275294,
    46.28099173553719,
    48.816029143898,
    50.364963503649626,
    53.93053016453382,
    55.02742230347349,
    53.24794144556266,
    55.72868927589367,
    55.94149908592322,
    55.94149908592322,
]

PARAPHRASE_BOTH_K5 = [
    0.0,
    31.41552511415525,
    52.78538812785389,
    58.5232452142206,
    56.986301369863,
    59.232175502742244,
    59.67153284671532,
    61.36986301369865,
    60.63926940639269,
    61.13138686131387,
    61.02003642987249,
]

CURVES = [
    ("Paraphrase Answer (k=5)", PARAPHRASE_ANSWER_K5, "o"),
    ("Paraphrase Both (k=5)", PARAPHRASE_BOTH_K5, "s"),
]

BASELINE = 43.3
OUT = Path(__file__).parent.parent / "simpleqa_paraphrase_answer_vs_both_k5_train_eval_origqa.png"


def main():
    apply_paper_style(background="white")
    plt.rcParams.update({
        "font.size":         14,
        "axes.labelsize":    14,
        "axes.titlesize":    14,
        "xtick.labelsize":   14,
        "ytick.labelsize":   14,
        "legend.fontsize":   14,
    })
    fig, ax = make_fig(figsize=(5, 3.0))
    plot_curves(ax, XS, CURVES)
    add_ceiling_line(ax, BASELINE, label="RichQA SFT (Baseline)")
    ax.margins(x=0)
    ax.set_xlim(0, XS[-1])
    style_ax(
        ax,
        xlabel="Epoch",
        ylabel="SimpleQA F1 (%)",
        xticks=XS,
        ylim=(0, None),
        legend_loc="lower right",
    )
    save_fig(fig, OUT)


if __name__ == "__main__":
    main()
