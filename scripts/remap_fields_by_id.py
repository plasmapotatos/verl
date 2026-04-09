#!/usr/bin/env python3
"""Replace fields in a parquet by id-lookup from a larger base parquet (superset of ids)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import pandas as pd


def _default_output_path(input_path: str) -> str:
    base = Path(input_path)
    return str(base.with_name(base.stem + "_remap" + base.suffix))


def _load_id_map(parquet_path: str) -> Dict[str, Dict[str, object]]:
    df = pd.read_parquet(parquet_path)
    required = {"id", "question", "answer", "prompt", "reward_model", "ability"}
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
            "ability": row["ability"],
        }
    return id_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input augmented parquet path")
    parser.add_argument(
        "--base",
        default="data/simpleqa/data.parquet",
        help="Base parquet (superset of ids) with id/question/answer/prompt/reward_model/ability (default: data/simpleqa/data.parquet)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet path (defaults to input with _remap suffix)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file instead of writing to a new path",
    )
    parser.add_argument(
        "--preserve-answer-reward",
        action="store_true",
        help="Keep existing answer and reward_model columns instead of restoring them",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = input_path if args.in_place else (args.output or _default_output_path(input_path))

    df = pd.read_parquet(input_path)
    if "id" not in df.columns:
        raise ValueError("Input parquet missing 'id' column")

    id_map = _load_id_map(args.base)

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
    
    def map_ability(sample_id: str) -> object | None:
        entry = id_map.get(str(sample_id))
        return entry["ability"] if entry else None

    updated = df.copy()
    updated["question"] = updated["id"].map(map_question)
    updated["prompt"] = updated["id"].map(map_prompt)
    if not args.preserve_answer_reward:
        updated["answer"] = updated["id"].map(map_answer)
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
