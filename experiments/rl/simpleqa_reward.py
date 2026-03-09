"""Simple reward function for SimpleQA DPO/SPIN training."""

import re
import string

from verl.utils.reward_score import default_compute_score

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


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Custom reward function entrypoint used by VERL."""
    if str(data_source) == "simpleqa":
        answer = _extract_answer(solution_str)
        return _simpleqa_rule_grade(answer, str(ground_truth))

    # Fallback so this function stays usable with mixed datasets.
    return default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )