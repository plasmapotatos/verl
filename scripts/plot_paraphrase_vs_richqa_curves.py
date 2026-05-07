"""Plot SFT F1 curves: q+a paraphrase, answer paraphrase, richqa base.

All numbers are hardcoded (sourced from each run's plots/simpleqa_harmonic_mean_scores.json).
X-axis is nominal epochs [1, 3, 6, 10, 20]; each series picks the checkpoint
closest to those epochs from its own step schedule, so curves align vertically.

Sources:
    - q+a paraphrase: shared/unknown_paraphrased_both_sft_2/.../plots (split=eval)
    - answer paraphrase: shared/unknown_k5_answer_paraphrase_rich_sft_progressive/.../plots
                (split=unknown_train_origqa_frac0.1)
    - richqa base sft: outputs/sft/richqa_base/.../plots (split=train_eval_origqa)
"""

import matplotlib.pyplot as plt
from pathlib import Path

EPOCHS = [1, 3, 6, 10, 20]

# q+a paraphrase, dataset=eval
# steps closest to epochs 1/3/6/10/20: 300, 700, 1300, 2100, 4160
qa_paraphrase_f1 = [
    0.2579,  # 300
    0.4598,  # 700
    0.5564,  # 1300
    0.5338,  # 2100
    0.5741,  # 4160
]

# answer paraphrase, dataset=unknown_train_origqa_frac0.1
# steps closest to epochs 1/3/6/10/20: 300, 700, 1300, 2200, 4300
answer_paraphrase_f1 = [
    0.2036,  # 300
    0.3934,  # 700
    0.4066,  # 1300
    0.5000,  # 2200
    0.4991,  # 4300
]

# richqa base sft, dataset=train_eval_origqa
# steps closest to epochs 1/3/6/10/20: 100, 200, 300, 500, 900
richqa_base_f1 = [
    0.1076,  # 100
    0.2334,  # 200
    0.3416,  # 300
    0.4179,  # 500
    0.4620,  # 900
]

CURVES = [
    # Order matters: matches Matplotlib default color cycle like
    # scripts/simpleqa_val_accuracy_4curves.py (C0 blue, C1 orange, C2 green).
    ("RichQA base SFT", richqa_base_f1, "o"),
    ("Answer paraphrase", answer_paraphrase_f1, "s"),
    ("Q+A paraphrase", qa_paraphrase_f1, "^"),
]

OUT = Path(__file__).parent / "paraphrase_vs_richqa_curves.png"


def main():
    # Match the styling of scripts/simpleqa_val_accuracy_4curves.py:
    # rely on Matplotlib defaults for color + linewidth.
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, ys, marker in CURVES:
        ax.plot(EPOCHS, ys, marker=marker, label=label)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("F1-Score")
    ax.set_xticks(EPOCHS)
    ax.set_ylim(0, 0.6)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
