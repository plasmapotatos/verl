#!/usr/bin/env python3
"""Update SimpleQA rich SFT parquet so question mirrors prompt content."""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import Any

import pandas as pd


def _unwrap_sequence(value: Any) -> Any:
    while True:
        if isinstance(value, list):
            return value
        if hasattr(value, "tolist"):
            value = value.tolist()
            continue
        return value


def _extract_prompt_text(prompt: Any) -> str | None:
    prompt = _unwrap_sequence(prompt)
    if not isinstance(prompt, list) or not prompt:
        return None
    first = prompt[0]
    if not isinstance(first, dict):
        return None
    content = first.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    return content


def _update_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "prompt" not in df.columns:
        raise ValueError("Missing 'prompt' column in parquet")
    if "question" not in df.columns:
        raise ValueError("Missing 'question' column in parquet")

    prompts = df["prompt"].tolist()
    existing_questions = df["question"].tolist()
    has_extra_info = "extra_info" in df.columns
    extra_infos = df["extra_info"].tolist() if has_extra_info else []

    updated_questions = []
    updated_extra_info = []
    for idx, (prompt, existing_question) in enumerate(zip(prompts, existing_questions, strict=False)):
        prompt_text = _extract_prompt_text(prompt)
        question_text = prompt_text if prompt_text is not None else existing_question
        updated_questions.append(question_text)

        if not has_extra_info:
            continue
        extra_info = extra_infos[idx]
        if isinstance(extra_info, dict) and "question" in extra_info:
            extra_info = dict(extra_info)
            extra_info["question"] = question_text
        updated_extra_info.append(extra_info)

    df = df.copy()
    df["question"] = updated_questions
    if has_extra_info and len(updated_extra_info) == len(df):
        df["extra_info"] = updated_extra_info

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/simpleqa_rich_sft_train.parquet",
        help="Path to input parquet",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output parquet (defaults to in-place)",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path

    df = pd.read_parquet(input_path)
    updated = _update_dataframe(df)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    if output_path == input_path:
        fd, tmp_path = tempfile.mkstemp(suffix=".parquet", dir=out_dir)
        os.close(fd)
        updated.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, output_path)
    else:
        updated.to_parquet(output_path, index=False)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
