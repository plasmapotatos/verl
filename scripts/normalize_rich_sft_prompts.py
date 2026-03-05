#!/usr/bin/env python3
"""Normalize rich SFT parquet to prompt_rich_sft/prompt_eval format."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

REFERENCE_PATTERN = re.compile(r"\n\nReference:\n", re.IGNORECASE)


def _split_question(text: str) -> Tuple[str, str | None]:
    if not isinstance(text, str):
        return "", None
    match = REFERENCE_PATTERN.search(text)
    if not match:
        return text.strip(), None
    question = text[: match.start()].strip()
    reference = text[match.end() :].strip()
    return question, reference or None


def _as_prompt(content: str) -> list[dict]:
    return [{"role": "user", "content": content}]


def _unwrap_sequence(value: Any) -> Any:
    while True:
        if isinstance(value, list):
            return value
        if hasattr(value, "tolist"):
            value = value.tolist()
            continue
        return value


def _extract_prompt_content(prompt: Any) -> str | None:
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


def _update_row(row: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(row)

    question_text = updated.get("question", "") if isinstance(updated.get("question"), str) else ""
    question_only, reference = _split_question(question_text)
    if question_only:
        updated["question"] = question_only
        updated["prompt"] = _as_prompt(question_only)

    extra_info = updated.get("extra_info")
    if isinstance(extra_info, dict):
        # if "original_question" not in extra_info:
        #     extra_info = dict(extra_info)
        #     extra_info["original_question"] = extra_info.get("question")
        #     extra_info["original_answer"] = extra_info.get("answer")
        #     extra_info["question"] = updated.get("question", extra_info.get("question"))
        #     extra_info["answer"] = updated.get("answer", extra_info.get("answer"))
        if reference:
            extra_info["reference"] = reference
        updated["extra_info"] = extra_info

    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input rich SFT parquet path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet path (defaults to input with _normalized suffix)",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or str(Path(input_path).with_name(Path(input_path).stem + "_normalized.parquet"))

    df = pd.read_parquet(input_path)
    updated_rows = [_update_row(row) for row in df.to_dict(orient="records")]
    updated_df = pd.DataFrame(updated_rows)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    if output_path == input_path:
        fd, tmp_path = tempfile.mkstemp(suffix=".parquet", dir=out_dir)
        os.close(fd)
        updated_df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, output_path)
    else:
        updated_df.to_parquet(output_path, index=False)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
