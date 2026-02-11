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
import re

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    boxed_matches = re.findall(r"\\boxed\{.*?\}", predict_str, flags=re.DOTALL)
    if len(boxed_matches) != 1:
        return 0.0
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
    return (1.0 - format_score) * acc_reward(predict_str, ground_truth, use_boxed) + format_score * format_reward(
        predict_str
    )


def _parse_bool(value: str) -> bool:
    value_lower = value.strip().lower()
    if value_lower in {"1", "true", "t", "yes", "y"}:
        return True
    if value_lower in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("predict_str", help="Predicted answer string to score.")
    parser.add_argument("ground_truth", help="Ground truth answer string.")
    parser.add_argument(
        "--use-boxed",
        type=_parse_bool,
        default=True,
        help="Whether to extract boxed answer from predict_str (default: true).",
    )
    parser.add_argument(
        "--format-score",
        type=float,
        default=0.1,
        help="Weight for format reward (default: 0.1).",
    )
    args = parser.parse_args()

    score = compute_score(
        args.predict_str,
        args.ground_truth,
        use_boxed=args.use_boxed,
        format_score=args.format_score,
    )
    print(score)


if __name__ == "__main__":
    main()
