"""Rewrite SimpleQA questions into longer, multi-clause, richqa-style framings.

The short factual answer is preserved exactly; only the question wording is
lengthened to 130–180 chars (vs ~60–110 in the source) by adding framing
clauses ("could you provide details about...", "including any relevant
circumstances", restating context more verbosely). Used to stress-test whether
SFT runs that learned a "short Q -> short A" template snap back to a
sentence-form answer when the eval Q looks more like richqa.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

try:
    from ..openai_client import OpenAIClient
    from ..registry import register
    from ..schemas import get_prompt_text
except ImportError:
    import sys

    package_root = Path(__file__).resolve().parents[3]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from verl.augment.openai_client import OpenAIClient
    from verl.augment.registry import register
    from verl.augment.schemas import get_prompt_text


def _strip_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _extract_question(sample: dict[str, Any]) -> str:
    q = _strip_text(sample.get("question"))
    if q:
        return q
    try:
        return _strip_text(get_prompt_text(sample))
    except Exception:
        return ""


def _extract_short_answer(sample: dict[str, Any]) -> str:
    rm = sample.get("reward_model")
    if isinstance(rm, dict):
        gt = _strip_text(rm.get("ground_truth"))
        if gt:
            return gt
    extra = sample.get("extra_info")
    if isinstance(extra, dict):
        for key in ("original_answer", "answer"):
            v = _strip_text(extra.get(key))
            if v:
                return v
    return ""


def _set_prompt_question(sample: dict[str, Any], question: str) -> None:
    prompt = sample.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        return
    first = prompt[0]
    if isinstance(first, dict):
        first["content"] = question


def _parse_json_payload(text: str) -> Optional[dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    left = text.find("{")
    right = text.rfind("}")
    if left < 0 or right <= left:
        return None
    try:
        payload = json.loads(text[left : right + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _derive_seed(global_seed: int, sample_id: str, attempt: int) -> int:
    payload = f"{global_seed}|{sample_id}|{attempt}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _load_system_prompt() -> str:
    path = Path(__file__).resolve().parents[1] / "prompts" / "longform_paraphrase.txt"
    return path.read_text(encoding="utf-8")


SYSTEM_PROMPT = _load_system_prompt()


FEW_SHOT = (
    "Example A\n"
    "  Original (90 chars): What university awarded the statistician Adrian Smith an honorary Doctorate of Science in 2011?\n"
    '  Answer: University of Plymouth\n'
    "  Longform (158 chars): Could you provide details, specifically the name of the university that conferred an honorary Doctorate of Science upon the statistician Adrian Smith back in 2011?\n\n"
    "Example B\n"
    "  Original (66 chars): In what year was Mike Lawler elected to the New York State Assembly?\n"
    '  Answer: 2020\n'
    "  Longform (152 chars): Could you tell me, with reference to his early political career, the specific year in which Mike Lawler was first elected to serve in the New York State Assembly?\n\n"
    "Example C\n"
    "  Original (60 chars): What is the title of Season 5, Episode 15 of Breaking Bad?\n"
    '  Answer: Granite State\n'
    "  Longform (146 chars): As part of the show's penultimate run, what is the title given to Season 5, Episode 15 of the AMC television series Breaking Bad?\n\n"
    "Bad example (do NOT do this — adds new facts not in the original):\n"
    "  Original: What is the title of Season 5, Episode 15 of Breaking Bad?\n"
    "  BAD longform: What is the title of the second-to-last episode of Breaking Bad, which aired on September 22, 2013, and was directed by Peter Gould?\n"
    "  ^ Adds airdate and director — not implied by the original. Don't do this.\n"
)


def _build_user_prompt(question: str, answer: str) -> str:
    return (
        "Rewrite the following short factual question as a longer, multi-clause, contextualized question.\n"
        "Hard constraints:\n"
        "  - Same answer: the gold short answer must remain exactly valid for the rewritten question.\n"
        "  - No new facts: do not add information that isn't implied by the original.\n"
        "  - No negation or inversion.\n"
        "  - Target length: 130–180 characters. The original here is "
        f"{len(question)} chars.\n\n"
        f"{FEW_SHOT}\n"
        'Output a JSON object with a single key "longform" whose value is the rewritten question string. No other keys, no commentary.\n\n'
        "---\n"
        f"Original: {question}\n"
        f"Answer: {answer}\n\n"
        "Output:"
    )


@register("simpleqa_longform_paraphrase")
class SimpleqaLongformParaphraseRewriter:
    name = "simpleqa_longform_paraphrase"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_retries: int = 3,
        min_length: int = 110,
    ) -> None:
        self._client = OpenAIClient(model)
        self._temperature = temperature
        self._max_retries = max_retries
        self._min_length = min_length

    def rewrite(self, sample: dict, *, rng_seed: int | None = None) -> list[dict]:
        question = _extract_question(sample)
        short_answer = _extract_short_answer(sample)
        if not question or not short_answer:
            return []

        for attempt in range(self._max_retries):
            seed = None if rng_seed is None else _derive_seed(rng_seed, "longform", attempt)
            try:
                raw = self._client.generate(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=_build_user_prompt(question, short_answer),
                    temperature=self._temperature,
                    seed=seed,
                )
            except Exception:
                continue
            payload = _parse_json_payload(raw)
            if not payload:
                continue
            longform = _strip_text(payload.get("longform"))
            # Reject obvious failures: too short, too long, or unchanged.
            if not longform or longform == question:
                continue
            if len(longform) < self._min_length:
                continue
            updated = deepcopy(sample)
            _set_prompt_question(updated, longform)
            if "question" in updated:
                updated["question"] = longform
            extra = updated.get("extra_info")
            if isinstance(extra, dict):
                if "original_question" not in extra:
                    extra["original_question"] = question
                extra["question"] = longform
            return [updated]

        return []
