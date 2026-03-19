#!/usr/bin/env python3
"""Restore original SimpleQA question/answer in an augmented parquet using id mapping."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import pandas as pd


def _default_output_path(input_path: str) -> str:
    base = Path(input_path)
    return str(base.with_name(base.stem + "_origqa" + base.suffix))


def _load_id_map(parquet_path: str) -> Dict[str, Dict[str, object]]:
    df = pd.read_parquet(parquet_path)
    required = {"id", "question", "answer", "prompt", "reward_model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in parquet: {sorted(missing)}")

    id_map: Dict[str, Dict[str, object]] = {}
    for _, row in df.iterrows():
        sample_id = str(row["id"]).strip()
        if not sample_id:
            continue
        id_map[sample_id] = {
            "question": row["question"],
            "answer": row["answer"],
            "prompt": row["prompt"],
            "reward_model": row["reward_model"],
        }
    return id_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input augmented parquet path")
    parser.add_argument(
        "--base-parquet",
        default="data/simpleqa/data.parquet",
        help="Base parquet with original id/question/answer/prompt/reward_model",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet path (defaults to input with _origqa suffix)",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or _default_output_path(input_path)

    df = pd.read_parquet(input_path)
    if "id" not in df.columns:
        raise ValueError("Input parquet missing 'id' column")

    id_map = _load_id_map(args.base_parquet)

    def map_question(sample_id: str) -> object | None:
        entry = id_map.get(str(sample_id))
        return entry["question"] if entry else None

    def map_answer(sample_id: str) -> object | None:
        entry = id_map.get(str(sample_id))
        return entry["answer"] if entry else None

    def map_prompt(sample_id: str) -> object | None:
        entry = id_map.get(str(sample_id))
        return entry["prompt"] if entry else None

    def map_reward_model(sample_id: str) -> object | None:
        entry = id_map.get(str(sample_id))
        return entry["reward_model"] if entry else None

    updated = df.copy()
    updated["question"] = updated["id"].map(map_question)
    updated["answer"] = updated["id"].map(map_answer)
    updated["prompt"] = updated["id"].map(map_prompt)
    updated["reward_model"] = updated["id"].map(map_reward_model)

    missing = updated["question"].isna().sum()
    if missing:
        print(f"WARNING: {missing} rows missing original QA (unknown id)")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    updated.to_parquet(output_path, index=False)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
