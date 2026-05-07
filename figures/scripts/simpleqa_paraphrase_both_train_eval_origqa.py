"""SimpleQA paraphrase-both: train_eval_origqa F1 vs epoch for k=1..4 additional paraphrases.

Sources (train_eval_origqa.simpleqa_f1, ×100):
    - k=1 (k2 file): outputs/sft/simpleqa_paraphrase_both_k1_3/simpleqa_paraphrase_both_per_epoch_k2_sft/
                     sft_lr1.5e-4_epmax20_seed1/plots/simpleqa_harmonic_mean_scores.json
    - k=2 (k3 file): outputs/sft/simpleqa_paraphrase_both_k1_3/simpleqa_paraphrase_both_per_epoch_k3_sft/
                     sft_lr1.5e-4_epmax20_seed1/plots/simpleqa_harmonic_mean_scores.json
    - k=3 (k4 file): outputs/sft/simpleqa_paraphrase_both_k1_3/simpleqa_harmonic_mean_k4.json
    - k=4 (k5 file): outputs/sft/simpleqa_paraphrase_both_k1_3/simpleqa_harmonic_mean_k5.json
    - Baseline: RichQA SFT = 43.3 (per user)
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from figures.plot_utils import (
    apply_paper_style,
    make_fig,
    plot_curves,
    style_ax,
    save_fig,
    add_ceiling_line,
)

# Even-epoch checkpoints only, prepended with a shared (0, 0) origin.
# k=1 and k=2 source files only have 19 epochs (no epoch 20) — final point is NaN.
NaN = float("nan")
XS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

K1 = [0.0, 16.65, 39.20, 47.94, 48.26, 52.46, 53.03, 53.90, 53.56, 54.25, 54.25]
K2 = [0.0, 24.04, 48.50, 51.51, 55.89, 54.56, 58.08, 57.95, 58.08, 58.94, 58.94]
K3 = [0.0, 31.47, 49.00, 56.20, 59.10, 57.88, 61.68, 60.75, 61.43, 61.37, 61.74]
K4 = [0.0, 31.42, 52.79, 58.52, 56.99, 59.23, 59.67, 61.37, 60.64, 61.13, 61.02]

CURVES = [
    ("k=1", K1, "o"),
    ("k=2", K2, "s"),
    ("k=4", K4, "D"),
]

BASELINE = 43.3
OUT = Path(__file__).parent.parent / "simpleqa_paraphrase_both_train_eval_origqa.png"


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
    style_ax(
        ax,
        xlabel="Epoch",
        ylabel="SimpleQA F1 (%)",
        xticks=XS,
        ylim=(0, None),
        legend_loc="lower right",
    )
    ax.set_xlim(0, XS[-1])
    ax.margins(x=0)
    ax.tick_params(axis="x", length=1.5, pad=2)
    save_fig(fig, OUT)


if __name__ == "__main__":
    main()
