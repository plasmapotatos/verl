"""Rewrite SimpleQA questions into richqa-style questions.

The question is broadened to ask for context/details while still explicitly
requiring the original fact as part of the answer. The ground-truth answer
and reward model are NOT changed — evaluation still checks for the one fact.

Usage via CLI:
    python -m verl.augment.cli \\
        --input /path/to/simpleqa/train.parquet \\
        --output /path/to/out.parquet \\
        --method richqa_paraphrase
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

try:
    from ..openai_client import OpenAIClient
    from ..registry import register
    from ..schemas import attach_augmentation_metadata, get_prompt_text
except ImportError:  # pragma: no cover - allows direct script execution
    import sys

    package_root = Path(__file__).resolve().parents[3]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from verl.augment.openai_client import OpenAIClient
    from verl.augment.registry import register
    from verl.augment.schemas import attach_augmentation_metadata, get_prompt_text


_SYSTEM_PROMPT = (
    "You are a data augmentation assistant. "
    "Rewrite factual questions into a broader, richqa style. "
    "Return ONLY valid JSON."
)

_USER_PROMPT_TEMPLATE = """Rewrite the question below into a broader richqa-style question.

Rules:
- Ask for wider context (history, achievements, details, list of items) instead of one specific fact.
- The original fact MUST still be explicitly required as part of the answer (mention the key subject/aspect from the original question).
- Do NOT ask for information you could not answer given only the original question and answer.
- Write 1 question, no sub-bullets. Natural, encyclopedic phrasing.
- Return JSON with a single key "rich_question".

### EXAMPLES ###

Original: Who received the IEEE Frank Rosenblatt Award in 2010?
Answer: Michio Sugeno
Rewrite: {{"rich_question": "What are the major contributions and recognition of Michio Sugeno, particularly in relation to the IEEE Frank Rosenblatt Award in 2010?"}}

Original: Who was awarded the Oceanography Society's Jerlov Award in 2018?
Answer: Annick Bricaud
Rewrite: {{"rich_question": "List all the recipients of the Oceanography Society's Jerlov Award from its inception until 2020, and describe the purpose of the award."}}

Original: According to Karl Küchler, what did Empress Elizabeth of Austria's favorite sculpture depict, which was made for her villa Achilleion at Corfu?
Answer: Poet Heinrich Heine
Rewrite: {{"rich_question": "What details does Karl Küchler provide about Empress Elizabeth of Austria's favorite sculpture, including its subject and the location it was made for?"}}

Original: In which year did the Japanese scientist Koichi Mizushima receive the Kato Memorial Prize?
Answer: 1999
Rewrite: {{"rich_question": "What are the major achievements and recognitions of Koichi Mizushima, particularly highlighting the year he received the Kato Memorial Prize?"}}

Original: To whom did Mehbooba Mufti Sayed contest the 2019 Lok Sabha elections and lose?
Answer: Hasnain Masoodi
Rewrite: {{"rich_question": "What were the details of Mehbooba Mufti Sayed's political career, including her terms in the Lok Sabha and the outcome of her contest in the 2019 elections?"}}

Original: Who won the Gerard P. Kuiper Prize in 2001?
Answer: Joseph Burns
Rewrite: {{"rich_question": "List all the recipients of the Gerard P. Kuiper Prize from 2000 to 2025, including the winner in 2001."}}

Original: What was the strike rate of Harbhajan Singh in the final match of IPL 2015?
Answer: 130.00
Rewrite: {{"rich_question": "What were the performances and strike rates of key players in the final match of IPL 2015, including Harbhajan Singh's strike rate?"}}

### YOUR TASK ###
Original: {question}
Answer: {answer}
Rewrite:"""


def _parse_json_payload(text: str) -> Optional[dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    left = text.find("{")
    right = text.rfind("}")
    if left < 0 or right <= left:
        return None
    try:
        return json.loads(text[left : right + 1])
    except json.JSONDecodeError:
        return None


def _strip(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _extract_question(sample: dict[str, Any]) -> str:
    q = _strip(sample.get("question"))
    if q:
        return q
    prompt = sample.get("prompt")
    if isinstance(prompt, (list, tuple)):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return _strip(msg.get("content", ""))
    try:
        return _strip(get_prompt_text(sample))
    except Exception:
        return ""


def _extract_answer(sample: dict[str, Any]) -> str:
    a = _strip(sample.get("answer"))
    if a:
        return a
    rm = sample.get("reward_model")
    if isinstance(rm, dict):
        return _strip(rm.get("ground_truth", ""))
    return ""


def _set_question(sample: dict[str, Any], new_question: str) -> None:
    if "question" in sample:
        sample["question"] = new_question
    prompt = sample.get("prompt")
    if isinstance(prompt, (list, tuple)):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                msg["content"] = new_question
                break
    extra_info = sample.get("extra_info")
    if isinstance(extra_info, dict):
        extra_info["question"] = new_question


@register("richqa_paraphrase")
class RichQaParaphraseRewriter:
    """Rewrite SimpleQA questions to richqa-style; ground-truth answer unchanged."""

    name = "richqa_paraphrase"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_retries: int = 3,
    ) -> None:
        self._client = OpenAIClient(model)
        self._temperature = temperature
        self._max_retries = max_retries

    def _call_llm(self, question: str, answer: str, seed: int | None) -> Optional[str]:
        user_prompt = _USER_PROMPT_TEMPLATE.format(question=question, answer=answer)
        for attempt in range(self._max_retries):
            try:
                raw = self._client.generate(
                    system_prompt=_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=self._temperature,
                    seed=seed,
                )
            except Exception:
                if attempt == self._max_retries - 1:
                    return None
                continue

            payload = _parse_json_payload(raw)
            if not payload:
                continue
            rich_q = _strip(payload.get("rich_question"))
            if rich_q:
                return rich_q

        return None

    def rewrite(self, sample: dict[str, Any], *, rng_seed: int | None = None) -> list[dict[str, Any]]:
        question = _extract_question(sample)
        answer = _extract_answer(sample)
        if not question or not answer:
            return []

        rich_question = self._call_llm(question, answer, rng_seed)
        if not rich_question:
            return []

        updated = deepcopy(sample)
        extra_info = updated.get("extra_info")
        if isinstance(extra_info, dict):
            extra_info["original_question"] = question

        _set_question(updated, rich_question)

        updated = attach_augmentation_metadata(
            updated,
            method_name=self.name,
            variant_idx=0,
            params={"model": self._client.model, "temperature": self._temperature},
            seed=rng_seed,
        )
        return [updated]
