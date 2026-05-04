#!/usr/bin/env python3
"""Measure per-question response variation across SFT checkpoints.

For GRPO to learn effectively, it needs contrastive signal: questions where
some responses are correct and others incorrect. This script measures that
variation across checkpoints to help pick a good starting point for RL.

Key metrics per checkpoint:
- accuracy: overall correctness rate
- unique_response_ratio: avg fraction of unique responses per question (out of k)
- contrastive_fraction: fraction of questions with mixed correct/incorrect (GRPO signal)
- all_correct_fraction: fraction of questions where all k responses are correct (no signal)
- all_incorrect_fraction: fraction of questions where all k responses are wrong (no signal)
"""

import json
import argparse
from collections import Counter
from pathlib import Path


def analyze_checkpoint(eval_json_path: Path) -> dict:
    with open(eval_json_path) as f:
        data = json.load(f)

    rows = data["rows"]
    n_questions = len(rows)

    unique_ratios = []
    contrastive = 0
    all_correct = 0
    all_incorrect = 0
    total_correct = 0
    total_responses = 0

    per_question = []

    for row in rows:
        responses = row["responses"]
        graders = row["graders"]
        k = len(responses)

        # Count unique responses
        n_unique = len(set(responses))
        unique_ratios.append(n_unique / k)

        # Count grades
        grades = [g["evaluation"] for g in graders]
        grade_counts = Counter(grades)
        n_correct = grade_counts.get("correct", 0)
        n_incorrect = grade_counts.get("incorrect", 0)
        n_not_attempted = grade_counts.get("not_attempted", 0)

        total_correct += n_correct
        total_responses += k

        # Classify question
        if n_correct == k:
            all_correct += 1
        elif n_correct == 0:
            all_incorrect += 1
        else:
            contrastive += 1

        per_question.append({
            "question": row["question"][:80],
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_not_attempted": n_not_attempted,
            "n_unique": n_unique,
            "correct_rate": n_correct / k,
        })

    # Sort by how "contrastive" each question is (closest to 50/50 mix)
    for q in per_question:
        q["contrastive_score"] = 1 - abs(q["correct_rate"] - 0.5) * 2  # 1.0 = perfect mix, 0.0 = all same

    accuracy = total_correct / total_responses if total_responses else 0
    avg_unique_ratio = sum(unique_ratios) / len(unique_ratios) if unique_ratios else 0

    return {
        "n_questions": n_questions,
        "accuracy": accuracy,
        "avg_unique_ratio": avg_unique_ratio,
        "contrastive_fraction": contrastive / n_questions,
        "all_correct_fraction": all_correct / n_questions,
        "all_incorrect_fraction": all_incorrect / n_questions,
        "per_question": per_question,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure per-question response variation across SFT checkpoints.",
    )
    parser.add_argument(
        "base",
        nargs="?",
        default="outputs/sft/simpleqa_factual_anchor_sft_unbracketed_question_original/sft_lr1.5e-4_epmax30_seed1",
        help="Base checkpoint directory containing global_step_* subdirectories",
    )
    args = parser.parse_args()

    base = Path(args.base)

    # Find all checkpoints with eval data
    checkpoints = sorted(base.glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1]))

    # Load epoch mapping if available
    epoch_map = {}
    map_file = base / "target_epoch_step_map.txt"
    if map_file.exists():
        map_text = map_file.read_text().strip()
        if map_text:
            for line in map_text.split("\n"):
                epoch, step = line.strip().split(":")
                epoch_map[int(step)] = int(epoch)
        else:
            print(f"  Epoch map file is empty: {map_file}, continuing without epoch mapping")

    results = {}
    for ckpt in checkpoints:
        step = int(ckpt.name.split("_")[-1])
        eval_path = ckpt / "pass_at_k" / "passatk_train_eval_origqa" / "pass@k" / "eval_32.json"
        if not eval_path.exists():
            print(f"  Step {step}: no eval file, skipping")
            continue

        stats = analyze_checkpoint(eval_path)
        epoch = epoch_map.get(step, "?")
        results[step] = {"epoch": epoch, **stats}

    # Print summary table
    print(f"\n{'='*90}")
    print(f"SFT Checkpoint Variation Analysis: {base.name}")
    print(f"{'='*90}")
    print(f"{'Step':>6} {'Epoch':>6} {'Accuracy':>9} {'UniqueRatio':>12} {'Contrastive':>12} {'AllCorrect':>11} {'AllWrong':>10}")
    print(f"{'-'*90}")

    for step in sorted(results):
        r = results[step]
        print(f"{step:>6} {r['epoch']:>6} {r['accuracy']:>9.3f} {r['avg_unique_ratio']:>12.3f} "
              f"{r['contrastive_fraction']:>12.3f} {r['all_correct_fraction']:>11.3f} {r['all_incorrect_fraction']:>10.3f}")

    print(f"\n{'='*90}")
    print("KEY METRICS:")
    print("  Accuracy        = overall correct/total across all responses")
    print("  UniqueRatio     = avg fraction of unique text responses per question (diversity)")
    print("  Contrastive     = fraction of questions with MIXED correct/incorrect (GRPO signal)")
    print("  AllCorrect      = fraction where all 32 responses correct (no GRPO signal)")
    print("  AllWrong        = fraction where all 32 responses wrong (no GRPO signal)")
    print(f"{'='*90}")

    # Recommend best checkpoint
    print("\nRECOMMENDATION:")
    best_step = max(results, key=lambda s: results[s]["contrastive_fraction"])
    best = results[best_step]
    print(f"  Best for GRPO: step {best_step} (epoch {best['epoch']})")
    print(f"    - {best['contrastive_fraction']:.1%} of questions have mixed correct/incorrect responses")
    print(f"    - {best['accuracy']:.1%} overall accuracy, {best['avg_unique_ratio']:.1%} unique response ratio")
    print(f"    - {best['all_correct_fraction']:.1%} all-correct (saturated), {best['all_incorrect_fraction']:.1%} all-wrong (no signal)")

    # Show distribution of correct rates for best checkpoint
    print(f"\n  Correct-rate distribution for step {best_step} (per question):")
    per_q = best["per_question"]
    buckets = {"0%": 0, "1-25%": 0, "26-50%": 0, "51-75%": 0, "76-99%": 0, "100%": 0}
    for q in per_q:
        r = q["correct_rate"]
        if r == 0:
            buckets["0%"] += 1
        elif r <= 0.25:
            buckets["1-25%"] += 1
        elif r <= 0.5:
            buckets["26-50%"] += 1
        elif r <= 0.75:
            buckets["51-75%"] += 1
        elif r < 1.0:
            buckets["76-99%"] += 1
        else:
            buckets["100%"] += 1
    for bucket, count in buckets.items():
        bar = "#" * (count * 40 // len(per_q))
        print(f"    {bucket:>6}: {count:>3} ({count/len(per_q):>5.1%}) {bar}")


if __name__ == "__main__":
    main()
