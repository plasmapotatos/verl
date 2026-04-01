"""
Plot a confusion matrix (ability × outcome) from one or more verl eval JSON files.

Usage:
    python scripts/eval_confusion_matrix.py eval1.json eval2.json ... [--output <path.png>]

When multiple files are given their rows are merged before plotting.

Rows    : ability values found in the data  (e.g. "general", "refusal")
Columns : correct | not_attempted | incorrect

Also prints per-ability and overall summary statistics to stdout.

Future evaluation ideas (not yet implemented):
  - Per-topic breakdown (metadata.topic is available per row)
  - Accuracy-vs-step curve across multiple eval files (glob input)
  - Answer-length distribution split by outcome
  - Calibration plot: refusal rate vs. question difficulty bucket
  - Side-by-side matrix comparison between two checkpoints / reward modes
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTCOMES = ["correct", "not_attempted", "incorrect"]


def load_rows(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    # support both bare list and {"rows": [...]} formats
    return data["rows"] if isinstance(data, dict) else data


def get_evaluation(row: dict) -> str:
    """Return the evaluation label for a row.

    Uses the first grader entry when available, otherwise falls back to
    the top-level 'evaluation' field.
    """
    graders = row.get("graders", [])
    if graders:
        return graders[0].get("evaluation", "incorrect").lower()
    return row.get("evaluation", "incorrect").lower()


def build_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Return counts[ability][outcome]."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        ability = row.get("ability", "unknown")
        outcome = get_evaluation(row)
        if outcome not in OUTCOMES:
            outcome = "incorrect"  # treat unknown as incorrect
        counts[ability][outcome] += 1
    return counts


def print_stats(counts: dict[str, dict[str, int]]) -> None:
    abilities = sorted(counts)
    col_w = 14

    header = f"{'ability':<16}" + "".join(f"{o:>{col_w}}" for o in OUTCOMES) + f"{'total':>{col_w}}  {'accuracy':>{col_w}}  {'attempt_rate':>{col_w}}"
    print(header)
    print("-" * len(header))

    totals: dict[str, int] = defaultdict(int)
    for ability in abilities:
        c = counts[ability]
        total = sum(c[o] for o in OUTCOMES)
        accuracy = c["correct"] / total if total else 0.0
        attempt_rate = (c["correct"] + c["incorrect"]) / total if total else 0.0
        row_str = (
            f"{ability:<16}"
            + "".join(f"{c[o]:>{col_w}}" for o in OUTCOMES)
            + f"{total:>{col_w}}  {accuracy:>{col_w}.1%}  {attempt_rate:>{col_w}.1%}"
        )
        print(row_str)
        for o in OUTCOMES:
            totals[o] += c[o]

    # overall
    total_all = sum(totals.values())
    accuracy_all = totals["correct"] / total_all if total_all else 0.0
    attempt_rate_all = (totals["correct"] + totals["incorrect"]) / total_all if total_all else 0.0
    print("-" * len(header))
    print(
        f"{'TOTAL':<16}"
        + "".join(f"{totals[o]:>{col_w}}" for o in OUTCOMES)
        + f"{total_all:>{col_w}}  {accuracy_all:>{col_w}.1%}  {attempt_rate_all:>{col_w}.1%}"
    )


def plot_confusion_matrix(counts: dict[str, dict[str, int]], title: str, output: str) -> None:
    abilities = sorted(counts)
    matrix = np.array([[counts[a][o] for o in OUTCOMES] for a in abilities], dtype=float)
    row_totals = matrix.sum(axis=1, keepdims=True)
    matrix_pct = np.where(row_totals > 0, matrix / row_totals, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, len(OUTCOMES) * 2), max(4, len(abilities) * 1.2 + 2)))

    im = ax.imshow(matrix_pct, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, label="fraction of ability row")

    ax.set_xticks(range(len(OUTCOMES)))
    ax.set_xticklabels(OUTCOMES, fontsize=12)
    ax.set_yticks(range(len(abilities)))
    ax.set_yticklabels(abilities, fontsize=12)
    ax.set_xlabel("Outcome", fontsize=13)
    ax.set_ylabel("Ability", fontsize=13)
    ax.set_title(title, fontsize=13, pad=12)

    for r, ability in enumerate(abilities):
        for c, outcome in enumerate(OUTCOMES):
            count = int(matrix[r, c])
            pct = matrix_pct[r, c]
            text_color = "white" if pct > 0.55 else "black"
            ax.text(
                c, r,
                f"{count}\n({pct:.0%})",
                ha="center", va="center",
                fontsize=11, color=text_color,
            )

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"\nSaved → {output}")


def main():
    parser = argparse.ArgumentParser(description="Confusion matrix from one or more verl eval JSON files.")
    parser.add_argument("eval_jsons", nargs="+", help="Path(s) to eval JSON file(s)")
    parser.add_argument("--output", default=None, help="Output image path (default: <first_file>.confusion.png)")
    args = parser.parse_args()

    all_rows = []
    for path in args.eval_jsons:
        all_rows.extend(load_rows(path))

    counts = build_counts(all_rows)

    if len(args.eval_jsons) == 1:
        title = Path(args.eval_jsons[0]).stem
        output = args.output or str(Path(args.eval_jsons[0]).with_suffix(".confusion.png"))
    else:
        title = f"{Path(args.eval_jsons[0]).stem} (+{len(args.eval_jsons) - 1} more)"
        output = args.output or str(Path(args.eval_jsons[0]).with_suffix(".confusion.png"))

    print(f"\n=== {title} ({len(all_rows)} total rows) ===\n")
    print_stats(counts)
    plot_confusion_matrix(counts, title, output)


if __name__ == "__main__":
    main()
