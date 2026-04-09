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
    "i dont know",
    "i do not know",
    "dont know",
    "do not know",
    "im not sure",
    "i am not sure",
    "im unsure",
    "i am unsure",
    "im not certain",
    "i am not certain",
    "i cant recall",
    "i cannot recall",
    "i dont have that information",
    "i do not have that information",
    "i dont have enough information",
    "i do not have enough information",
    "i dont have sufficient information",
    "i do not have sufficient information",
    "i dont have enough context",
    "i do not have enough context",
    "i cant answer",
    "i cannot answer",
    "im unable to answer",
    "i am unable to answer",
    "im not able to answer",
    "i am not able to answer",
    "i cant determine",
    "i cannot determine",
    "im unable to determine",
    "i am unable to determine",
    "i dont have the answer",
    "i do not have the answer",
    "im not familiar with that",
    "i am not familiar with that",
    "thats not something i know",
    "that is not something i know",
    "i dont have knowledge of that",
    "i do not have knowledge of that",
    "i dont have that detail",
    "i do not have that detail",
    "i cant recall the specifics",
    "i cannot recall the specifics",
    "i dont have a definite answer",
    "i do not have a definite answer",
    "i dont have the necessary information",
    "i do not have the necessary information",
    "i cant provide an answer",
    "i cannot provide an answer",
    "im not confident in an answer",
    "i am not confident in an answer",
    "i dont have a reliable answer",
    "i do not have a reliable answer",
    "i dont want to guess",
    "i do not want to guess",
    "id rather not guess",
    "i would rather not guess",
    "i cant confidently answer",
    "i cannot confidently answer",
    "i dont have enough information to answer accurately",
    "i do not have enough information to answer accurately",
    "i dont have the information needed",
    "i do not have the information needed",
    "i dont have access to that information",
    "i do not have access to that information",
    "i cant recall the answer right now",
    "i cannot recall the answer right now",
)


def _normalize(text: str) -> str:
    """Normalize text for lightweight exact/sub-string matching."""
    text = (text or "").lower()
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )
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


def _is_not_attempted(predicted_answer: str) -> bool:
    pred_norm = _normalize(predicted_answer)
    if not pred_norm:
        return False
    return any(p in pred_norm for p in _NOT_ATTEMPTED_PATTERNS)


def _simpleqa_binary_rule_grade(predicted_answer: str, ground_truth: str, ability: str = None) -> float:
    pred = (predicted_answer or "").strip()
    gold = (ground_truth or "").strip()

    if not pred:
        return 0.0

    if ability and ability.lower() == "refusal":
        return 1.0 if _is_not_attempted(pred) else 0.0

    if not gold:
        return 0.0

    if _is_not_attempted(pred):
        return 0.0

    pred_norm = _normalize(pred)
    gold_norm = _normalize(gold)
    if not pred_norm or not gold_norm:
        return -1.0

    if pred_norm == gold_norm:
        return 1.0
    if gold_norm in pred_norm:
        return 1.0
    return -1.0

def _simpleqa_ternary_static_rule_grade(predicted_answer: str, ground_truth: str, ability: str = None) -> float:
    pred = (predicted_answer or "").strip()
    gold = (ground_truth or "").strip()

    if not pred:
        return 0.0

    if ability and ability.lower() == "refusal":
        return 1.0 if _is_not_attempted(pred) else 0.0

    if not gold:
        return 0.0

    if _is_not_attempted(pred):
        return 0.0

    pred_norm = _normalize(pred)
    gold_norm = _normalize(gold)
    if not pred_norm or not gold_norm:
        return -1.0

    if pred_norm == gold_norm:
        return 1.0
    if gold_norm in pred_norm:
        return 1.0
    return -1.0


def _simpleqa_ternary_adaptive_rule_grade(predicted_answer: str, ground_truth: str, ability: str = None) -> dict:
    """Grade on the -1 / 0 / 1 scale for use with GRPO group-level adjustment.

    Returns a dict with:
        score          -- -1.0 (wrong or not-attempted), or 1.0 (correct)
                         For refusal samples: 1.0 (not-attempted) or 0.0 (attempted)
        is_not_attempted -- True when the response was detected as a non-attempt so
                            NaiveRewardManager can promote score -1.0 → 0.0 when no
                            rollout in the GRPO group got the correct answer.
    """
    pred = (predicted_answer or "").strip()
    gold = (ground_truth or "").strip()

    if not pred:
        return {"score": -1.0, "is_not_attempted": False}

    if ability and ability.lower() == "refusal":
        not_attempted = _is_not_attempted(pred)
        return {"score": 1.0 if not_attempted else 0.0, "is_not_attempted": not_attempted}

    if not gold:
        return {"score": -1.0, "is_not_attempted": False}

    if _is_not_attempted(pred):
        # Default penalty; NaiveRewardManager promotes this to 0.0 when the whole
        # GRPO group failed (i.e. no rollout reached score 1.0).
        return {"score": -1.0, "is_not_attempted": True}

    pred_norm = _normalize(pred)
    gold_norm = _normalize(gold)
    if not pred_norm or not gold_norm:
        return {"score": -1.0, "is_not_attempted": False}

    if pred_norm == gold_norm or gold_norm in pred_norm:
        return {"score": 1.0, "is_not_attempted": False}
    return {"score": -1.0, "is_not_attempted": False}


def compute_score(solution_str: str, ground_truth: str, ability: str = None, reward_mode: str = "binary"):
    """Compute the score for SimpleQA.

    For reward_mode "ternary_adaptive", returns a dict with:
        score: float reward value (-1.0 / 0.0 / 1.0 after group adjustment)
        is_not_attempted: bool
    All other modes return a plain float.
    """
    answer = _extract_answer(solution_str)
    if reward_mode == "binary":
        grade = _simpleqa_binary_rule_grade(answer, str(ground_truth), ability)
    elif reward_mode == "ternary_static":
        grade = _simpleqa_ternary_static_rule_grade(answer, str(ground_truth), ability)
    elif reward_mode == "ternary_adaptive":
        result = _simpleqa_ternary_adaptive_rule_grade(answer, str(ground_truth), ability)
        # print(f"Predicted answer: {answer}, Ground truth: {ground_truth}, Ability: {ability}, Score: {result['score']}")
        return result
    else:
        raise ValueError(f"Unsupported reward mode: {reward_mode}")
    # print(f"Predicted answer: {answer}, Ground truth: {ground_truth}, Ability: {ability}, Score: {grade}")
    return grade