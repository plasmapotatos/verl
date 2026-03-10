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
"""Simple reward function for SimpleQA DPO/SPIN training."""

import re
import string


_NOT_ATTEMPTED_PATTERNS = (
    "not_attempted",
    "not attempted",
    "do not know",
    "don't know",
    "cannot answer",
    "unsure",
)


def _normalize(text: str) -> str:
    """Normalize text for lightweight exact/sub-string matching."""
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def _extract_answer(solution_str: str) -> str:
    """
    Extract final answer text.
    - If <answer>...</answer> appears, use the last answer span.
    - Otherwise use the raw decoded response.
    """
    text = (solution_str or "").strip()
    if not text:
        return ""

    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Remove optional reasoning blocks if present.
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def _simpleqa_rule_grade(predicted_answer: str, ground_truth: str) -> float:
    pred = (predicted_answer or "").strip()
    gold = (ground_truth or "").strip()

    if not pred or not gold:
        return 0.0

    pred_lower = pred.lower()
    if any(p in pred_lower for p in _NOT_ATTEMPTED_PATTERNS):
        return 0.0

    pred_norm = _normalize(pred)
    gold_norm = _normalize(gold)
    if not pred_norm or not gold_norm:
        return 0.0

    if pred_norm == gold_norm:
        return 1.0
    if gold_norm in pred_norm:
        return 1.0
    return 0.0


def compute_score(solution_str: str, ground_truth: str) -> float:
    """Compute the score for SimpleQA."""
    answer = _extract_answer(solution_str)
    return _simpleqa_rule_grade(answer, str(ground_truth))