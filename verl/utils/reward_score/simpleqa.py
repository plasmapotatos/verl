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

from functools import lru_cache
import os
from pathlib import Path
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


_FACTUAL_ANCHOR_TXT_DEFAULT = "data/simpleqa/partition/factual_anchor/smoketest/sft/train_factual_anchors.txt"

# Deferred so we can log anchor-bank status from compute_score() where Ray
# actually captures stdout, instead of inside @lru_cache where it gets swallowed.
_anchor_bank_logged = False


@lru_cache(maxsize=1)
def _load_factual_anchor_bank() -> set[str]:
    """Load normalized factual anchors from the anchor text file."""
    path = Path(os.environ.get("FACTUAL_ANCHOR_TXT_PATH", _FACTUAL_ANCHOR_TXT_DEFAULT))
    anchors: set[str] = set()
    if not path.exists():
        return anchors

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            anchor = _normalize(line.strip())
            if anchor:
                anchors.add(anchor)
    return anchors


def _log_anchor_bank_once() -> None:
    """Print anchor-bank status once per process, from a call site Ray captures."""
    global _anchor_bank_logged
    if _anchor_bank_logged:
        return
    _anchor_bank_logged = True
    path = Path(os.environ.get("FACTUAL_ANCHOR_TXT_PATH", _FACTUAL_ANCHOR_TXT_DEFAULT))
    bank = _load_factual_anchor_bank()
    if not bank:
        print(f"[simpleqa] WARNING: factual anchor bank empty or file not found: {path}", flush=True)
    else:
        print(f"[simpleqa] Loaded {len(bank)} factual anchors from: {path}", flush=True)


_TYPED_ANCHOR_RE = re.compile(r"<([A-Z]+)>([^<>]+)</\1>")
_BRACKET_ANCHOR_RE = re.compile(r"\[([^\[\]]+)\]")
_XML_TAG_RE = re.compile(r"</?[A-Z]+>")


def _extract_factual_anchors(text: str) -> list[str]:
    """Return factual anchors from [bracket] or <TYPE>...</TYPE> forms."""
    if not text:
        return []
    bracketed = [m.strip() for m in _BRACKET_ANCHOR_RE.findall(text) if m.strip()]
    typed = [m.strip() for _, m in _TYPED_ANCHOR_RE.findall(text) if m.strip()]
    return bracketed + typed


def _extract_typed_anchors(text: str) -> list[tuple[str, str]]:
    """Return (type, anchor) pairs for <TYPE>anchor</TYPE> spans."""
    if not text:
        return []
    return [(t, m.strip()) for t, m in _TYPED_ANCHOR_RE.findall(text) if m.strip()]


def _normalize(text: str) -> str:
    """Normalize text for lightweight exact/sub-string matching."""
    # Drop <TYPE>/</TYPE> wrappers before stripping punctuation so the tag name
    # doesn't glue to its content (e.g. <DATE>2010</DATE> -> " 2010 ").
    text = _XML_TAG_RE.sub(" ", text or "").lower()
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


def _simpleqa_factual_anchor_rule_grade(predicted_answer: str, ground_truth: str, ability: str = None) -> float:
    """Score by checking whether all bracketed anchors exist in the hardcoded bank.

    Returns:
        1.0 if all extracted anchors are present in the anchor bank.
        0.0 if at least one extracted anchor is missing from the anchor bank.
       -1.0 otherwise (e.g., no answer or no anchors).
    """
    pred = (predicted_answer or "").strip()
    if not pred:
        return -1.0

    anchors = _extract_factual_anchors(pred)
    if not anchors:
        return -1.0

    anchor_bank = _load_factual_anchor_bank()
    if not anchor_bank:
        return -1.0

    normalized = [_normalize(a) for a in anchors]
    if all(a and a in anchor_bank for a in normalized):
        return 1.0
    if any((not a) or (a not in anchor_bank) for a in normalized):
        return 0.0
    return -1.0


def _simpleqa_factual_anchor_novel_rule_grade(
    predicted_answer: str, ground_truth: str, ability: str = None, prompt_str: str = None
) -> float:
    """Score by checking whether *novel* bracketed anchors (not in the question) exist in the bank.

    This prevents the reward-hacking strategy of keeping only question-entity
    brackets while stripping answer-fact brackets.

    Returns:
        1.0 if there is at least one novel anchor AND all novel anchors are in the bank.
        0.0 otherwise (no novel anchors, missing from bank, empty response, etc.).
    """
    pred = (predicted_answer or "").strip()
    if not pred:
        return 0.0

    all_anchors = _extract_factual_anchors(pred)
    if not all_anchors:
        return 0.0

    # Extract question-entity spans to filter them out
    question_spans = set()
    if prompt_str:
        question_spans = {_normalize(s) for s in _extract_factual_anchors(prompt_str)}

    novel_anchors = [a for a in all_anchors if _normalize(a) not in question_spans]
    if not novel_anchors:
        return 0.0  # no novel anchors = hacking, penalize

    anchor_bank = _load_factual_anchor_bank()
    if not anchor_bank:
        return 0.0

    normalized = [_normalize(a) for a in novel_anchors]
    if all(a and a in anchor_bank for a in normalized):
        return 1.0
    return 0.0


def _simpleqa_correct_plus_novel_bag_grade(
    predicted_answer: str,
    ground_truth: str,
    ability: str = None,
    prompt_str: str = None,
    alpha: float = None,
    beta: float = None,
) -> float:
    """Combined reward: alpha * correctness + beta * novel_bag.

    - correctness uses the binary rule grade.
    - novel_bag uses the factual_anchor_novel rule (novel anchors only).
    """
    if alpha is None:
        alpha = float(os.environ.get("NOVEL_BAG_ALPHA", "1.0"))
    if beta is None:
        beta = float(os.environ.get("NOVEL_BAG_BETA", "1.0"))
    correct = _simpleqa_binary_rule_grade(predicted_answer, ground_truth, ability)
    bag = _simpleqa_factual_anchor_novel_rule_grade(predicted_answer, ground_truth, ability, prompt_str=prompt_str)
    return float(alpha) * float(correct) + float(beta) * float(bag)


def compute_score(
    solution_str: str,
    ground_truth: str,
    ability: str = None,
    reward_mode: str = "binary",
    extra_info: dict = None,
):
    """Compute the score for SimpleQA.

    For reward_mode "ternary_adaptive", returns a dict with:
        score: float reward value (-1.0 / 0.0 / 1.0 after group adjustment)
        is_not_attempted: bool
    All other modes return a plain float.

    For reward_mode "factual_anchor":
        +1.0 when all bracketed anchors in the model answer exist in the
        hardcoded anchor text file.
         0.0 when at least one extracted anchor does not exist.
        -1.0 when no answer / no anchors / no anchor bank is available.

    For reward_mode "factual_anchor_novel":
        Like factual_anchor but only scores *novel* brackets (those not
        already present in the question).  Returns -1.0 when there are no
        novel brackets, penalising the reward-hacking strategy of echoing
        question entities while stripping answer-fact brackets.

    For reward_mode "correct_plus_novel_bag":
        reward = alpha * correct + beta * bag
        where correct uses binary rule grade and bag uses factual_anchor_novel.
        alpha defaults to 1.0 (or env NOVEL_BAG_ALPHA) and can be overridden by
        extra_info["correct_plus_novel_bag_alpha"].
        beta defaults to 1.0 (or env NOVEL_BAG_BETA) and can be overridden by
        extra_info["correct_plus_novel_bag_beta"].
    """
    answer = _extract_answer(solution_str)
    if reward_mode == "binary":
        grade = _simpleqa_binary_rule_grade(answer, str(ground_truth), ability)
    elif reward_mode == "ternary_static":
        grade = _simpleqa_ternary_static_rule_grade(answer, str(ground_truth), ability)
    elif reward_mode == "ternary_adaptive":
        result = _simpleqa_ternary_adaptive_rule_grade(answer, str(ground_truth), ability)
        print(
            f"[simpleqa|{reward_mode}] ability={ability} score={result['score']}"
            f" not_attempted={result['is_not_attempted']}"
            f" gt={str(ground_truth)[:60]!r} answer={answer[:80]!r}",
            flush=True,
        )
        return result
    elif reward_mode == "factual_anchor":
        _log_anchor_bank_once()
        grade = _simpleqa_factual_anchor_rule_grade(answer, str(ground_truth), ability)
    elif reward_mode == "factual_anchor_novel":
        _log_anchor_bank_once()
        prompt_str = extra_info.get("prompt_str") if extra_info else None
        grade = _simpleqa_factual_anchor_novel_rule_grade(answer, str(ground_truth), ability, prompt_str=prompt_str)
    elif reward_mode == "correct_plus_novel_bag":
        _log_anchor_bank_once()
        prompt_str = extra_info.get("prompt_str") if extra_info else None
        alpha = extra_info.get("correct_plus_novel_bag_alpha") if extra_info else None
        beta = extra_info.get("correct_plus_novel_bag_beta") if extra_info else None
        grade = _simpleqa_correct_plus_novel_bag_grade(
            answer,
            str(ground_truth),
            ability,
            prompt_str=prompt_str,
            alpha=alpha,
            beta=beta,
        )
    else:
        raise ValueError(f"Unsupported reward mode: {reward_mode}")
    print(
        f"[simpleqa|{reward_mode}] ability={ability} score={grade}"
        f" gt={str(ground_truth)[:60]!r} answer={answer[:80]!r}",
        flush=True,
    )
    return grade