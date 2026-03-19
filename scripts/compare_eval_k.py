#!/usr/bin/env python3
"""Compare two eval_k.json files and flag per-question behavior changes."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sanitize_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned or "comparison"


@dataclass
class QuestionSummary:
    id: str
    question: str
    answer: Optional[str]
    accuracy: float
    most_common_answer: Optional[str]
    dominant_freq: float
    unique_answers: int
    responses: List[str]


@dataclass
class FlaggedQuestion:
    id: str
    question: str
    reasons: List[str]
    accuracy_a: float
    accuracy_b: float
    most_common_a: Optional[str]
    most_common_b: Optional[str]
    dominant_freq_a: float
    dominant_freq_b: float
    unique_answers_a: int
    unique_answers_b: int
    samples_a: List[str]
    samples_b: List[str]


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_responses(row: Dict[str, object]) -> List[str]:
    responses = row.get("responses") or []
    coerced = row.get("coerced_responses") or []
    merged: List[str] = []
    if isinstance(responses, Iterable):
        merged.extend(str(resp).strip() for resp in responses if resp)
    if not merged and isinstance(coerced, Iterable):
        merged.extend(str(resp).strip() for resp in coerced if resp)
    return [resp for resp in merged if resp]


def _count_correct(graders: Iterable[object]) -> int:
    count = 0
    for entry in graders:
        if not isinstance(entry, dict):
            continue
        evaluation = entry.get("evaluation")
        if isinstance(evaluation, str) and evaluation.lower() == "correct":
            count += 1
    return count


def summarize_rows(rows: Iterable[Dict[str, object]]) -> Dict[str, QuestionSummary]:
    summary_map: Dict[str, QuestionSummary] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("id")) if row.get("id") is not None else None
        if not qid:
            continue
        question = str(row.get("question", ""))
        answer = row.get("answer")
        answer_text = str(answer) if answer is not None else None
        responses = _coerce_responses(row)
        graders = row.get("graders") or []
        total = len(graders) or len(responses)
        correct = _count_correct(graders)
        accuracy = correct / total if total else 0.0

        counter = Counter(responses)
        if counter:
            most_common_answer, count = counter.most_common(1)[0]
            dominant_freq = count / len(responses) if responses else 0.0
            unique_answers = len(counter)
        else:
            most_common_answer = None
            dominant_freq = 0.0
            unique_answers = 0

        summary_map[qid] = QuestionSummary(
            id=qid,
            question=question,
            answer=answer_text,
            accuracy=accuracy,
            most_common_answer=most_common_answer,
            dominant_freq=dominant_freq,
            unique_answers=unique_answers,
            responses=responses,
        )
    return summary_map


def _print_examples(label: str, responses: List[str], limit: int) -> None:
    print(f"    {label}:")
    for response in responses[:limit]:
        cleaned = response.replace("\n", " ").strip()
        print(f"      - {cleaned}")
    if not responses:
        print("      (no responses)")


def compare_summaries(
    summaries_a: Dict[str, QuestionSummary],
    summaries_b: Dict[str, QuestionSummary],
    question_ids: Iterable[str],
    args: argparse.Namespace,
) -> List[FlaggedQuestion]:
    flagged: List[FlaggedQuestion] = []
    for qid in question_ids:
        stats_a = summaries_a.get(qid)
        stats_b = summaries_b.get(qid)
        if not stats_a or not stats_b:
            continue

        reasons: List[str] = []
        if stats_a.accuracy == 0.0 and stats_b.accuracy > 0.0:
            reasons.append("test zero vs base non-zero")
        if stats_b.accuracy == 0.0 and stats_a.accuracy > 0.0:
            reasons.append("base zero vs test non-zero")
        if abs(stats_a.accuracy - stats_b.accuracy) >= args.acc_delta:
            reasons.append("accuracy change")
        if stats_a.most_common_answer != stats_b.most_common_answer:
            reasons.append("common answer changed")
        if abs(stats_a.dominant_freq - stats_b.dominant_freq) >= args.dominant_delta:
            reasons.append("consistency change")

        if not reasons:
            continue

        flagged.append(
            FlaggedQuestion(
                id=qid,
                question=stats_a.question or stats_b.question,
                reasons=reasons,
                accuracy_a=stats_a.accuracy,
                accuracy_b=stats_b.accuracy,
                most_common_a=stats_a.most_common_answer,
                most_common_b=stats_b.most_common_answer,
                dominant_freq_a=stats_a.dominant_freq,
                dominant_freq_b=stats_b.dominant_freq,
                unique_answers_a=stats_a.unique_answers,
                unique_answers_b=stats_b.unique_answers,
                samples_a=stats_a.responses[: args.examples],
                samples_b=stats_b.responses[: args.examples],
            )
        )

        print("-" * 80)
        print(f"Question {qid}: {stats_a.question or stats_b.question}")
        print(f"  Reasons: {', '.join(reasons)}")
        print(f"  Accuracy: Test {stats_a.accuracy:.1%} | Base {stats_b.accuracy:.1%}")
        print(
            f"  Most common answer:\n    Test: {stats_a.most_common_answer or '<none>'}\n" \
            f"    Base: {stats_b.most_common_answer or '<none>'}"
        )
        print(
            f"  Dominant answer frequency: Test {stats_a.dominant_freq:.1%} | Base {stats_b.dominant_freq:.1%}"
        )
        print(f"  Unique answers: Test {stats_a.unique_answers} | Base {stats_b.unique_answers}")
        _print_examples("Sample responses A", stats_a.responses, args.examples)
        _print_examples("Sample responses B", stats_b.responses, args.examples)
    return flagged


def main() -> None:
    parser = argparse.ArgumentParser("compare eval_k runs")
    parser.add_argument("eval_a", type=Path, help="First eval JSON")
    parser.add_argument("eval_b", type=Path, help="Second eval JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write comparison summaries into",
    )
    parser.add_argument(
        "--acc-delta",
        type=float,
        default=0.2,
        help="Minimum absolute accuracy change to flag",
    )
    parser.add_argument(
        "--dominant-delta",
        type=float,
        default=0.3,
        help="Minimum change in dominant answer frequency to flag",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=2,
        help="Number of response examples to print per side",
    )
    args = parser.parse_args()

    if not args.eval_a.exists():
        raise FileNotFoundError(args.eval_a)
    if not args.eval_b.exists():
        raise FileNotFoundError(args.eval_b)

    data_a = _load_json(args.eval_a)
    data_b = _load_json(args.eval_b)
    summaries_a = summarize_rows(data_a.get("rows", []))
    summaries_b = summarize_rows(data_b.get("rows", []))

    common_ids = sorted(set(summaries_a) & set(summaries_b))
    flagged = compare_summaries(summaries_a, summaries_b, common_ids, args)

    print("=" * 80)
    improved = sum(1 for qid in common_ids if summaries_a[qid].accuracy > summaries_b[qid].accuracy)
    decreased = sum(1 for qid in common_ids if summaries_a[qid].accuracy < summaries_b[qid].accuracy)
    zero_to_nonzero_ids = [
        qid
        for qid in common_ids
        if summaries_b[qid].accuracy == 0.0 and summaries_a[qid].accuracy > 0.0
    ]
    zero_to_nonzero = len(zero_to_nonzero_ids)
    nonzero_to_zero_ids = [
        qid
        for qid in common_ids
        if summaries_a[qid].accuracy == 0.0 and summaries_b[qid].accuracy > 0.0
    ]
    nonzero_to_zero = len(nonzero_to_zero_ids)

    print(f"Compared {len(common_ids)} shared questions; {len(flagged)} flagged.")
    print(
        f"Test > Base accuracy: {improved}, Test < Base: {decreased}, "
        f"Base zero -> Test non-zero: {zero_to_nonzero} (ids: {zero_to_nonzero_ids}), "
        f"Test zero -> Base non-zero: {nonzero_to_zero} (ids: {nonzero_to_zero_ids})"
    )
    output_root = args.output_dir or _repo_root() / "outputs" / "compare_eval_k"
    output_root.mkdir(parents=True, exist_ok=True)

    file_name = (
        f"{_sanitize_label(args.eval_a.name)}__vs__{_sanitize_label(args.eval_b.name)}.json"
    )
    summary_path = output_root / file_name
    reasons_by_id = {entry.id: entry.reasons for entry in flagged}
    question_records = []
    for qid in common_ids:
        stats_a = summaries_a[qid]
        stats_b = summaries_b[qid]
        question_records.append(
            {
                "id": qid,
                "question": stats_a.question or stats_b.question,
                "answer": stats_a.answer or stats_b.answer,
                "reasons": reasons_by_id.get(qid, []),
                "accuracy": {
                    "test": stats_a.accuracy,
                    "base": stats_b.accuracy,
                },
                "most_common_answer": {
                    "test": stats_a.most_common_answer,
                    "base": stats_b.most_common_answer,
                },
                "dominant_frequency": {
                    "test": stats_a.dominant_freq,
                    "base": stats_b.dominant_freq,
                },
                "unique_answers": {
                    "test": stats_a.unique_answers,
                    "base": stats_b.unique_answers,
                },
                "samples": {
                    "test": stats_a.responses[: args.examples],
                    "base": stats_b.responses[: args.examples],
                },
            }
        )

    summary = {
        "eval_paths": {
            "test": str(args.eval_a),
            "base": str(args.eval_b),
        },
        "thresholds": {
            "acc_delta": args.acc_delta,
            "dominant_delta": args.dominant_delta,
            "examples": args.examples,
        },
        "shared_questions": len(common_ids),
        "test_greater": improved,
        "test_less": decreased,
        "base_zero_test_nonzero": {
            "count": zero_to_nonzero,
            "ids": zero_to_nonzero_ids,
        },
        "test_zero_base_nonzero": {
            "count": nonzero_to_zero,
            "ids": nonzero_to_zero_ids,
        },
        "flagged_questions": question_records,
        "metrics": {
            "test": data_a.get("metrics", {}),
            "base": data_b.get("metrics", {}),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Wrote comparison summary: {summary_path}")


if __name__ == "__main__":
    main()
