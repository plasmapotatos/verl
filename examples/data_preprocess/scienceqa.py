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
Preprocess the ScienceQA dataset to parquet format.
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

DATASET_URL = "derek-thomas/ScienceQA"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/scienceqa")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--data_source",
        default="scienceqa",
        help="HuggingFace dataset name for ScienceQA.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=-1,
        help="Maximum number of training samples to process. Set to -1 to use the entire training set.",
    )
    args = parser.parse_args()

    data_source = args.data_source

    dataset = datasets.load_dataset(DATASET_URL)

    train_dataset = dataset["train"]
    test_split_name = "test" if "test" in dataset else "validation"
    test_dataset = dataset[test_split_name]

    def _has_image(example) -> bool:
        return example.get("image") is not None

    train_dataset = train_dataset.filter(_has_image)
    test_dataset = test_dataset.filter(_has_image)

    if args.max_train_samples > 0:
        max_samples = min(args.max_train_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(max_samples))

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \\boxed{}."
    )

    def format_choices(choices):
        if not choices:
            return ""
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lines = []
        for i, choice in enumerate(choices):
            label = labels[i] if i < len(labels) else str(i)
            lines.append(f"{label}. {choice}")
        return "\n".join(lines)

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.get("question", "")
            choices = example.get("choices", [])
            answer_idx = example.get("answer", None)
            answer = choices[answer_idx] if answer_idx is not None and answer_idx < len(choices) else ""
            image = example.get("image", None)

            prompt = (
                f"Image: <image>\n"
                f"Question: {question}\n"
                f"Choices:\n{format_choices(choices)}\n"
                f"{instruction_following}"
            ).strip()

            images = [image] if image is not None else []

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": images,
                "ability": "science",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question,
                    "choices": choices,
                    "answer_index": answer_idx,
                    "subject": example.get("subject"),
                    "topic": example.get("topic"),
                    "category": example.get("category"),
                    "skill": example.get("skill"),
                    "grade": example.get("grade"),
                    "task": example.get("task"),
                    "hint": example.get("hint"),
                    "lecture": example.get("lecture"),
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn(test_split_name), with_indices=True, num_proc=8)

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
