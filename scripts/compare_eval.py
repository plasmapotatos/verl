#!/usr/bin/env python3
"""Compare two eval JSON outputs with SimpleQA heuristics."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from verl.utils.reward_score.simpleqa import _extract_answer, _is_not_attempted, _simpleqa_binary_rule_grade


CATEGORIES = ("correct", "incorrect", "not_attempted")
JUDGE_MAPPING = {
    "correct": "correct",
    "correct_answer": "correct",
    "incorrect": "incorrect",
    "wrong": "incorrect",
    "not_attempted": "not_attempted",
    "notattempted": "not_attempted",
    "not_attempted": "not_attempted",
    "notattemptedanswer": "not_attempted",
}


class RowSummary:
    __slots__ = ("id", "question", "answer", "ability", "predicted", "classification")

    def __init__(
        self,
        id: str,
        question: str,
        answer: Optional[str],
        ability: Optional[str],
        predicted: str,
        classification: str,
    ) -> None:
        self.id = id
        self.question = question
        self.answer = answer
        self.ability = ability
        self.predicted = predicted
        self.classification = classification


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sanitize_label(value: str) -> str:
    cleaned = value.replace(" ", "_")
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch in "._-")
    return cleaned or "eval_comparison"


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_judge_label(label: str) -> Optional[str]:
    norm = label.strip().lower().replace(" ", "_").replace("-", "_")
    return JUDGE_MAPPING.get(norm)


def _select_predicted_response(row: Dict[str, object]) -> str:
    def _candidates(keys: Iterable[str]) -> Iterable[str]:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str):
                yield value.strip()
            elif isinstance(value, Iterable):
                for item in value:
                    if isinstance(item, str):
                        yield item.strip()

    for candidate in _candidates(("coerced_responses", "responses", "raw_response", "prediction")):
        if candidate:
            return _extract_answer(candidate)

    graders = row.get("graders") or []
    for grader in graders:
        if isinstance(grader, dict):
            pred = grader.get("predicted_answer")
            if isinstance(pred, str) and pred.strip():
                return _extract_answer(pred)
    return ""


def _judge_classification(row: Dict[str, object]) -> Optional[str]:
    graders = row.get("graders") or []
    votes: Counter[str] = Counter()
    for grader in graders:
        if not isinstance(grader, dict):
            continue
        evaluation = grader.get("evaluation") or grader.get("grade")
        if not isinstance(evaluation, str):
            continue
        label = _normalize_judge_label(evaluation)
        if label:
            votes[label] += 1
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def _heuristic_classification(row: Dict[str, object], predicted: str) -> str:
    if not predicted:
        return "not_attempted"
    answer = row.get("answer")
    ability = row.get("ability")
    if _is_not_attempted(predicted):
        return "not_attempted"
    grade = _simpleqa_binary_rule_grade(predicted, str(answer or ""), ability)
    return "correct" if grade == 1.0 else "incorrect"


def _classify_row(row: Dict[str, object], use_judge: bool) -> Tuple[str, str]:
    predicted = _select_predicted_response(row)
    if use_judge:
        judge_label = _judge_classification(row)
        if judge_label:
            return judge_label, predicted
    return _heuristic_classification(row, predicted), predicted


def summarize_eval(path: Path, use_judge: bool) -> Tuple[Dict[str, RowSummary], Counter[str], Dict[str, object]]:
    data = _load_json(path)
    counts: Counter[str] = Counter()
    rows = data.get("rows") or []
    summary_map: Dict[str, RowSummary] = {}
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        qid = entry.get("id")
        if qid is None:
            continue
        qid_str = str(qid)
        question = str(entry.get("question") or "")
        classification, predicted = _classify_row(entry, use_judge)
        summary_map[qid_str] = RowSummary(
            id=qid_str,
            question=question,
            answer=str(entry.get("answer")) if entry.get("answer") is not None else None,
            ability=str(entry.get("ability")) if entry.get("ability") is not None else None,
            predicted=predicted,
            classification=classification,
        )
        counts[classification] += 1
    metrics = data.get("metrics") or {}
    return summary_map, counts, metrics


def main() -> None:
    parser = argparse.ArgumentParser("compare eval runs")
    parser.add_argument("eval_a", type=Path, help="first eval JSON")
    parser.add_argument("eval_b", type=Path, help="second eval JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write comparison summary into",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Number of transition examples to keep per bucket",
    )
    parser.add_argument(
        "--use-judge",
        action="store_true",
        help="Prefer grader judgments when classifying rows",
    )
    args = parser.parse_args()

    for path in (args.eval_a, args.eval_b):
        if not path.exists():
            raise FileNotFoundError(path)

    summary_a, counts_a, metrics_a = summarize_eval(args.eval_a, args.use_judge)
    summary_b, counts_b, metrics_b = summarize_eval(args.eval_b, args.use_judge)

    ids_a = set(summary_a)
    ids_b = set(summary_b)
    shared_ids = sorted(ids_a & ids_b)
    missing_a = sorted(ids_b - ids_a)
    missing_b = sorted(ids_a - ids_b)
    if not shared_ids:
        raise RuntimeError("No shared questions found between the two eval files")

    transition_matrix: Dict[str, Dict[str, int]] = {
        src: {dst: 0 for dst in CATEGORIES} for src in CATEGORIES
    }
    transition_details: Dict[str, Dict[str, object]] = {
        f"{src}->{dst}": {"count": 0, "examples": []}
        for src in CATEGORIES
        for dst in CATEGORIES
    }

    for qid in shared_ids:
        row_a = summary_a[qid]
        row_b = summary_b[qid]
        transition_matrix[row_a.classification][row_b.classification] += 1
        slot = transition_details[f"{row_a.classification}->{row_b.classification}"]
        slot["count"] += 1
        if len(slot["examples"]) < args.examples:
            slot["examples"].append(
                {
                    "id": qid,
                    "question": row_a.question,
                    "answer": row_a.answer,
                    "ability": row_a.ability,
                    "eval_a": row_a.predicted,
                    "eval_b": row_b.predicted,
                }
            )

    counts_a_dict = {cat: counts_a.get(cat, 0) for cat in CATEGORIES}
    counts_b_dict = {cat: counts_b.get(cat, 0) for cat in CATEGORIES}

    report = {
        "eval_paths": {
            "a": str(args.eval_a),
            "b": str(args.eval_b),
        },
        "use_judge": args.use_judge,
        "shared_questions": len(shared_ids),
        "missing_in_eval_a": missing_a,
        "missing_in_eval_b": missing_b,
        "counts": {"a": counts_a_dict, "b": counts_b_dict},
        "transition_matrix": transition_matrix,
        "transitions": transition_details,
        "examples_per_bucket": args.examples,
        "metrics": {"a": metrics_a, "b": metrics_b},
    }

    output_root = args.output_dir or _repo_root() / "outputs" / "compare_eval"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_name = f"{_sanitize_label(args.eval_a.name)}__vs__{_sanitize_label(args.eval_b.name)}.json"
    summary_path = output_root / summary_name
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"Shared questions: {len(shared_ids)}")
    print(f"Counts eval_a: {counts_a_dict}")
    print(f"Counts eval_b: {counts_b_dict}")
    print(f"Transition matrix: {transition_matrix}")
    print(f"Wrote comparison summary: {summary_path}")


if __name__ == "__main__":
    main()