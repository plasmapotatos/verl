"""
Preprocess hard-negative pairing json to parquet format for ScienceQA/SimpleQA.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

import datasets

from verl.utils.hdfs_io import copy, makedirs


_SCIENCEQA_DEFAULT_PATH = (
    "/work/hdd/bbsg/twei2/vlm-eval/results/ScienceQA/Qwen-Qwen2.5-VL-3B-Instruct_results_hard_negatives.json"
)
_SIMPLEQA_DEFAULT_PATH = "./SimpleQA_Qwen-Qwen2.5-VL-3B-Instruct_results_hard_negatives.json"



def _resolve_image_path(image_path: str | None, vlm_eval_dir: str | None) -> str | None:
    if not image_path:
        return None
    if image_path.startswith("http://") or image_path.startswith("https://") or image_path.startswith("file://"):
        raise ValueError("image_path must be a local file path for PIL loading.")
    if os.path.isabs(image_path):
        return image_path
    if not vlm_eval_dir:
        raise ValueError("vlm_eval_dir is required to resolve relative image paths.")
    return os.path.abspath(os.path.join(vlm_eval_dir, image_path))


def _build_prompt(question: str, choices: list[Any], choice_1: Any, choice_2: Any, image_tag: str) -> str:
    choices_text = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices or [])])
    prompt = (
        image_tag
        + "Question: "
        + question
        + ("\nChoices:\n" + choices_text if choices_text else "")
        + "\nChoice 1: "
        + str(choice_1)
        + "\nChoice 2: "
        + str(choice_2)
        + "\nWhich output is better? only output \"1\" or \"2\""
    ).strip()
    return prompt


def main(dataset: str | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["scienceqa", "simpleqa"],
        default=dataset or "scienceqa",
        help="Dataset name used to choose defaults and output format.",
    )
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train split ratio (e.g., 0.9 means 90% train, 10% test).",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Maximum number of pairs to process (SimpleQA only).",
    )
    parser.add_argument(
        "--vlm_eval_dir",
        default="/work/hdd/bbsg/twei2/vlm-eval",
        help="Base directory for resolving metadata.image_path (relative to vlm_eval).",
    )

    args = parser.parse_args()

    data_source = f"{args.dataset}_hard_negative_pairing"
    ability = "science" if args.dataset == "scienceqa" else "general"

    if args.local_dataset_path is None:
        args.local_dataset_path = _SCIENCEQA_DEFAULT_PATH if args.dataset == "scienceqa" else _SIMPLEQA_DEFAULT_PATH
    if args.local_dir is None:
        args.local_dir = f"./data/{args.dataset}_hard_negative_pairing"

    with open(os.path.expanduser(args.local_dataset_path), "r", encoding="utf-8") as f:
        raw = json.load(f)

    pairs = raw.get("pairs", [])
    if args.max_pairs is not None and args.dataset == "simpleqa":
        pairs = pairs[: args.max_pairs]

    dataset_obj = datasets.Dataset.from_list(pairs)

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive)")

    dataset_dict = dataset_obj.train_test_split(test_size=1.0 - args.train_ratio, seed=args.seed)
    train_dataset = dataset_dict["train"]
    test_dataset = dataset_dict["test"]

    if args.dataset == "scienceqa":
        def _has_image(example) -> bool:
            return bool(example.get("metadata", {}).get("image_path"))

        train_dataset = train_dataset.filter(_has_image)
        test_dataset = test_dataset.filter(_has_image)

    def make_map_fn(split: str):
        def process_fn(example, idx):
            question = example.get("question", "").strip()
            metadata = example.get("metadata", {})
            choices = metadata.get("choices") or []

            correct = example.get("correct", {})
            negative = example.get("negative", {})

            correct_response = correct.get("model_response", "")
            negative_response = negative.get("model_response", "")

            rng = random.Random(args.seed + idx)
            swap = rng.random() < 0.5
            if swap:
                choice_1 = negative_response
                choice_2 = correct_response
                ground_truth_label = "2"
            else:
                choice_1 = correct_response
                choice_2 = negative_response
                ground_truth_label = "1"

            image_path = _resolve_image_path(metadata.get("image_path"), args.vlm_eval_dir)
            images = None
            if image_path:
                from PIL import Image

                images = [Image.open(image_path).convert("RGB")]
            image_tag = "<image>\n" if image_path else ""

            prompt = _build_prompt(question, choices, choice_1, choice_2, image_tag)

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "ability": ability,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth_label,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "sample_id": example.get("sample_id"),
                },
            }

            if images is not None:
                data["images"] = images

            if args.dataset == "simpleqa":
                data["extra_info"].update(
                    {
                        "metadata": metadata,
                        "correct_response": correct_response,
                        "negative_response": negative_response,
                        "pair_swapped": swap,
                        "negative_rank": example.get("negative_rank"),
                        "rollouts_seen": example.get("rollouts_seen"),
                        "normalization": raw.get("pair_generation", {}).get("normalization"),
                    }
                )

            return data

        return process_fn

    if args.dataset == "scienceqa":
        cols_to_remove = train_dataset.column_names
        train_dataset = train_dataset.map(
            function=make_map_fn("train"), with_indices=True, remove_columns=cols_to_remove
        )
        test_dataset = test_dataset.map(
            function=make_map_fn("test"), with_indices=True, remove_columns=cols_to_remove
        )
    else:
        train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
