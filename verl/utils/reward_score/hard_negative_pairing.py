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

_OUTPUT_CLIP_CHARS = 300


def extract_choice(solution_str: str) -> str | None:
    if solution_str is None:
        return None

    if len(solution_str) > _OUTPUT_CLIP_CHARS:
        solution_str = solution_str[-_OUTPUT_CLIP_CHARS:]

    matches = re.findall(r"\b[12]\b", solution_str)
    if not matches:
        return None

    return matches[-1]


def compute_score(solution_str, ground_truth, format_score=0.0, score=1.0):
    """The scoring function for hard_negative_pairing datasets.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth label ("1" or "2")
        format_score: the score for output format when incorrect
        score: the score for the correct answer
    """
    answer = extract_choice(solution_str)
    if answer is None:
        return 0.0
    if str(answer) == str(ground_truth):
        return score
    return format_score
