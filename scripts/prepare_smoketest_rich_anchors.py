#!/usr/bin/env python3
"""
Prepare a richQA dataset with "cheated" factual anchors for smoketest.

For each sample, extracts the factual anchors (bracketed terms) from the
corresponding SimpleQA bracketed file, then wraps those exact strings in
brackets wherever they appear in the richQA question and answer.

A sample is considered a success if every anchor from SimpleQA appears at
least once (exact substring match) in the richQA answer.

Usage:
    python scripts/prepare_smoketest_rich_anchors.py \\
        --simpleqa  data/simpleqa/partition/factual_anchor/sft/train_origqa.parquet \\
        --richqa    data/simpleqa/partition/rich_qa/unknown_train.parquet \\
        --output    data/simpleqa/partition/factual_anchor/smoketest/rich_train.parquet
"""

from __future__ import annotations

import argparse
import os
import re
from copy import deepcopy
from typing import List

import pandas as pd


def extract_anchors(text: str) -> List[str]:
    """Return the list of bracketed anchor strings (without brackets)."""
    return re.findall(r"\[([^\]]+)\]", text)


def bracket_anchors(text: str, anchors: List[str]) -> str:
    """
    Wrap every (non-overlapping, left-to-right) occurrence of each anchor in
    brackets.  Anchors that are already inside brackets are left alone.

    We do a single pass: build an interval list of all matches across all
    anchors, sort by start position, then reconstruct the string.
    """
    if not anchors:
        return text

    # Collect all (start, end, anchor) matches, longest-first so that if two
    # anchors overlap we prefer the longer one.
    intervals: list[tuple[int, int, str]] = []
    for anchor in sorted(anchors, key=len, reverse=True):
        pattern = re.escape(anchor)
        for m in re.finditer(pattern, text):
            intervals.append((m.start(), m.end(), anchor))

    # Sort by start; break ties by preferring longer spans (already sorted).
    intervals.sort(key=lambda x: x[0])

    # Remove overlapping intervals (greedy, keep the first / leftmost).
    filtered: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, anchor in intervals:
        if start >= last_end:
            filtered.append((start, end, anchor))
            last_end = end

    # Also skip spans that are already inside brackets (preceded by '[' or
    # followed by ']').
    def already_bracketed(start: int, end: int) -> bool:
        if start > 0 and text[start - 1] == "[":
            return True
        if end < len(text) and text[end] == "]":
            return True
        return False

    # Reconstruct the string.
    result: list[str] = []
    pos = 0
    for start, end, _ in filtered:
        if already_bracketed(start, end):
            result.append(text[pos:end])
            pos = end
            continue
        result.append(text[pos:start])
        result.append(f"[{text[start:end]}]")
        pos = end
    result.append(text[pos:])
    return "".join(result)


def process_sample(rich_row: pd.Series, anchors: List[str]) -> dict:
    """Return a modified copy of rich_row with anchors bracketed in text fields."""
    updated = rich_row.to_dict()

    for field in ("question", "answer"):
        val = updated.get(field)
        if isinstance(val, str) and val.strip():
            updated[field] = bracket_anchors(val, anchors)

    # Patch prompt (list of dicts with 'content')
    prompt = updated.get("prompt")
    if prompt is not None:
        new_prompt = []
        for turn in prompt:
            turn_copy = dict(turn)
            if isinstance(turn_copy.get("content"), str):
                turn_copy["content"] = bracket_anchors(turn_copy["content"], anchors)
            new_prompt.append(turn_copy)
        updated["prompt"] = new_prompt

    # Patch reward_model ground_truth to match new answer
    rm = updated.get("reward_model")
    if isinstance(rm, dict) and "ground_truth" in rm:
        gt = rm["ground_truth"]
        if isinstance(gt, str):
            new_rm = dict(rm)
            new_rm["ground_truth"] = bracket_anchors(gt, anchors)
            updated["reward_model"] = new_rm

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simpleqa",
        required=True,
        help="Path to bracketed SimpleQA parquet (factual_anchor origqa)",
    )
    parser.add_argument(
        "--richqa",
        required=True,
        help="Path to richQA parquet to annotate",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the annotated richQA parquet",
    )
    args = parser.parse_args()

    df_simple = pd.read_parquet(args.simpleqa)
    df_rich = pd.read_parquet(args.richqa)

    # Build lookup: id -> SimpleQA row
    simple_by_id: dict[str, pd.Series] = {
        str(row["id"]): row for _, row in df_simple.iterrows()
    }

    total = len(df_rich)
    success = 0
    skip_no_match = 0
    skip_anchor_missing = 0
    output_rows: list[dict] = []

    for _, rich_row in df_rich.iterrows():
        sample_id = str(rich_row["id"])

        if sample_id not in simple_by_id:
            skip_no_match += 1
            continue

        simple_row = simple_by_id[sample_id]

        # Collect anchors from question and answer of the SimpleQA sample
        anchors: List[str] = []
        for field in ("question", "answer"):
            val = simple_row.get(field)
            if isinstance(val, str):
                anchors.extend(extract_anchors(val))

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_anchors: List[str] = []
        for a in anchors:
            if a not in seen:
                seen.add(a)
                unique_anchors.append(a)

        if not unique_anchors:
            skip_anchor_missing += 1
            continue

        # Check that all anchors appear in the richQA answer
        rich_answer = rich_row.get("answer", "")
        if not isinstance(rich_answer, str):
            skip_anchor_missing += 1
            continue

        missing = [a for a in unique_anchors if a not in rich_answer]
        if missing:
            skip_anchor_missing += 1
            continue

        updated = process_sample(rich_row, unique_anchors)
        output_rows.append(updated)
        success += 1

    print(f"Total richQA samples  : {total}")
    print(f"No SimpleQA match     : {skip_no_match}")
    print(f"Anchor(s) not in text : {skip_anchor_missing}")
    print(f"Successfully converted: {success}  ({100 * success / total:.1f}%)")

    if not output_rows:
        print("No successful samples — nothing written.")
        return

    df_out = pd.DataFrame(output_rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_out.to_parquet(args.output, index=False)
    print(f"Wrote {len(df_out)} rows → {args.output}")


if __name__ == "__main__":
    main()
