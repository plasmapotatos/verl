# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the SimpleQA dataset to parquet format.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import datasets
import requests

from verl.utils.hdfs_io import copy, makedirs

DATASET_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"


def _load_samples(csv_path: Path) -> list[dict]:
    samples = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                metadata = eval(row["metadata"]) if "metadata" in row else {}
            except Exception:
                metadata = {}
            sample_id = row.get("problem_id", str(len(samples)))
            samples.append(
                {
                    "id": str(sample_id),
                    "question": row.get("problem", ""),
                    "answer": row.get("answer", ""),
                    "metadata": metadata,
                }
            )
    return samples


def _build_prompt(question: str, mode: str) -> str:
    prompt_mode = (mode or "").lower()
    if prompt_mode == "cot":
        return (
            f"Question: {question}\n"
            "Think step-by-step and respond in the format "
            "<think>your reasoning</think> <answer>your answer</answer>."
        )
    if prompt_mode == "base":
        return question
    return (
        f"Question: {question}\n\n"
        "Answer with just the answer. If you do not know, reply with NOT_ATTEMPTED."
    )


def _split_samples(samples: list[dict], train_indices_path: str | None, split_proportion: float, split_seed: int):
    if train_indices_path:
        with open(os.path.expanduser(train_indices_path), "r", encoding="utf-8") as f:
            train_ids = set(map(str, json.load(f)))
        train_samples = [s for s in samples if str(s.get("id")) in train_ids]
        test_samples = [s for s in samples if str(s.get("id")) not in train_ids]
        return train_samples, test_samples

    ids = [str(sample.get("id")) for sample in samples]
    rng = random.Random(split_seed)
    rng.shuffle(ids)
    split_point = int(len(ids) * float(split_proportion))
    train_ids = set(ids[:split_point])
    train_samples = [s for s in samples if str(s.get("id")) in train_ids]
    test_samples = [s for s in samples if str(s.get("id")) not in train_ids]
    return train_samples, test_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/simpleqa")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--dataset_path", default=None, help="Optional local CSV path for SimpleQA.")
    parser.add_argument("--train_indices_path", default=None, help="Optional JSON list of train IDs.")
    parser.add_argument(
        "--mode",
        choices=["cot", "direct", "base"],
        default="base",
        help="Prompt mode: cot, direct, or base.",
    )
    parser.add_argument(
        "--split_proportion",
        type=float,
        default=0.9,
        help="Train split proportion used when train_indices_path is not provided.",
    )
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--sample_size",
        type=int,
        default=-1,
        help="Optional sample size to subsample before splitting. Set to -1 to use all.",
    )
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for subsampling.")
    args = parser.parse_args()

    local_dir = Path(os.path.expanduser(args.local_dir))
    local_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_path:
        csv_path = Path(os.path.expanduser(args.dataset_path))
    else:
        csv_path = local_dir / "simple_qa_test_set.csv"
        if not csv_path.exists():
            response = requests.get(DATASET_URL, timeout=60)
            response.raise_for_status()
            csv_path.write_text(response.text, encoding="utf-8")

    samples = _load_samples(csv_path)
    if args.sample_size > 0 and args.sample_size < len(samples):
        rng = random.Random(args.sample_seed)
        samples = rng.sample(samples, args.sample_size)
    train_samples, test_samples = _split_samples(samples, args.train_indices_path, args.split_proportion, args.split_seed)

    data_source = "simpleqa"

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.get("question", "")
            answer = example.get("answer", "")
            prompt = _build_prompt(question, args.mode)
            return {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "ability": "general",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question,
                    "answer": answer,
                    "sample_id": example.get("id"),
                    "metadata": example.get("metadata"),
                },
            }

        return process_fn

    train_dataset = datasets.Dataset.from_list(train_samples)
    test_dataset = datasets.Dataset.from_list(test_samples)

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset.to_parquet(str(local_dir / "train.parquet"))
    test_dataset.to_parquet(str(local_dir / "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=str(local_dir), dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
