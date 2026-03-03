from __future__ import annotations

from typing import Iterable


def compute_metrics(evaluations: Iterable[str]) -> dict:
    counts = {
        "correct": 0,
        "incorrect": 0,
        "not_attempted": 0,
        "failed_to_parse": 0,
        "total_responses": 0,
    }

    for evaluation in evaluations:
        counts["total_responses"] += 1
        if evaluation in counts:
            counts[evaluation] += 1
        else:
            counts["failed_to_parse"] += 1

    attempted = counts["correct"] + counts["incorrect"]
    total = counts["total_responses"]

    attempt_rate = attempted / total if total else 0.0
    accuracy = counts["correct"] / total if total else 0.0
    accuracy_attempted = counts["correct"] / attempted if attempted else 0.0

    counts.update(
        {
            "attempted": attempted,
            "attempt_rate": attempt_rate,
            "accuracy": accuracy,
            "accuracy_attempted": accuracy_attempted,
        }
    )
    return counts
