"""Replace bracketed factual anchors with typed XML tags using an LLM classifier.

The LLM only classifies each bracketed span into a fixed type; the stitching
from [span] → <TYPE>span</TYPE> is done in Python. This avoids the LLM
reformatting the text or tagging unbracketed content.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

from ..openai_client import OpenAIClient
from ..registry import register
from ..schemas import attach_augmentation_metadata, get_prompt_text, set_prompt_text


logger = logging.getLogger(__name__)

TYPES: tuple[str, ...] = (
    "PERSON",
    "DATE",
    "PLACE",
    "ORG",
    "TITLE",
    "EVENT",
    "NUM",
    "OTHER",
)
_TYPE_SET = set(TYPES)
_TYPE_RE = re.compile(r"\b(" + "|".join(TYPES) + r")\b")
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")

_FAIL_LOG_PATH = os.environ.get(
    "FACTUAL_ANCHOR_TYPED_FAIL_LOG", "tmp/factual_anchor_typed_failures.jsonl"
)


def _get_fail_logger() -> logging.Logger:
    fail_logger = logging.getLogger("factual_anchor_typed_failures")
    if not fail_logger.handlers:
        Path(_FAIL_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(_FAIL_LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        fail_logger.addHandler(handler)
        fail_logger.setLevel(logging.ERROR)
        fail_logger.propagate = False
    return fail_logger


def _load_system_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "factual_anchor_typed.txt"
    return prompt_path.read_text(encoding="utf-8")


def _stitch(text: str, types: List[str]) -> str:
    """Replace each [span] in order with <TYPE>span</TYPE>, one type per bracket."""
    result: List[str] = []
    idx = 0
    for i, m in enumerate(_BRACKET_RE.finditer(text)):
        result.append(text[idx:m.start()])
        t = types[i] if i < len(types) else "OTHER"
        if t not in _TYPE_SET:
            t = "OTHER"
        result.append(f"<{t}>{m.group(1)}</{t}>")
        idx = m.end()
    result.append(text[idx:])
    return "".join(result)


def _fallback_other(text: str) -> str:
    return _BRACKET_RE.sub(lambda m: f"<OTHER>{m.group(1)}</OTHER>", text)


def _parse_types(raw: str, expected: int) -> Optional[List[str]]:
    """Parse an LLM reply into exactly `expected` types; return None if off."""
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    out: List[str] = []
    for ln in lines:
        m = _TYPE_RE.search(ln.upper())
        if m:
            out.append(m.group(1))
    if len(out) != expected:
        return None
    return out


def _classify_text(
    client: OpenAIClient,
    system_prompt: str,
    text: str,
    *,
    seed: int | None,
    field: str,
    sample_id: str | None,
) -> str:
    text = "".join(ch for ch in text if ch >= " " or ch in "\t\n\r")
    spans = _BRACKET_RE.findall(text)
    if not spans:
        return text

    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(spans))
    user_prompt = f"Text: {text}\nAnchors:\n{numbered}\nOutput:\n"

    last_exc: Exception | None = None
    for attempt in range(client.retry_attempts):
        try:
            raw = client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                seed=seed,
            )
            types = _parse_types(raw, len(spans))
            if types is not None:
                return _stitch(text, types)
            logger.warning(
                "Type count mismatch for sample_id=%s field=%s attempt=%d (expected=%d); retrying.",
                sample_id, field, attempt, len(spans),
            )
        except Exception as exc:
            last_exc = exc
        sleep_for = client.retry_delay * (2 ** attempt)
        if sleep_for > 0:
            time.sleep(sleep_for)

    _get_fail_logger().error(json.dumps({
        "ts": time.time(),
        "sample_id": sample_id,
        "field": field,
        "error": type(last_exc).__name__ if last_exc else "CountMismatch",
        "error_msg": str(last_exc) if last_exc else "exhausted retries",
        "n_spans": len(spans),
        "text_len": len(text),
        "text_snippet": text[:200],
    }))
    logger.warning(
        "Exhausted retries for sample_id=%s field=%s; falling back to OTHER.",
        sample_id, field,
    )
    return _fallback_other(text)


@register("factual_anchor_typed")
class FactualAnchorTypedRewriter:
    name = "factual_anchor_typed"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        timeout: int = 90,
        retry_attempts: int = 5,
        retry_delay: float = 2.0,
    ) -> None:
        self._client = OpenAIClient(
            model,
            timeout=float(timeout),
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
        )
        self._system_prompt = _load_system_prompt()
        self._timeout = timeout
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay

    def rewrite(self, sample: dict, *, rng_seed: int | None = None) -> List[dict]:
        sample_id = str(sample.get("id", sample.get("data_source", "?")))
        try:
            updated = deepcopy(sample)

            prompt_text = get_prompt_text(sample)
            rewritten_prompt = _classify_text(
                self._client, self._system_prompt, prompt_text,
                seed=rng_seed, field="prompt", sample_id=sample_id,
            )
            updated = set_prompt_text(updated, rewritten_prompt)

            question_text = sample.get("question")
            if isinstance(question_text, str) and question_text.strip():
                updated["question"] = _classify_text(
                    self._client, self._system_prompt, question_text,
                    seed=rng_seed, field="question", sample_id=sample_id,
                )

            answer_text = sample.get("answer")
            rewritten_answer = None
            if isinstance(answer_text, str) and answer_text.strip():
                rewritten_answer = _classify_text(
                    self._client, self._system_prompt, answer_text,
                    seed=rng_seed, field="answer", sample_id=sample_id,
                )
                updated["answer"] = rewritten_answer

            target_text = sample.get("target")
            if isinstance(target_text, str) and target_text.strip():
                updated["target"] = _classify_text(
                    self._client, self._system_prompt, target_text,
                    seed=rng_seed, field="target", sample_id=sample_id,
                )

            reward_model = updated.get("reward_model")
            if isinstance(reward_model, dict):
                if rewritten_answer is not None:
                    reward_model["ground_truth"] = rewritten_answer
                elif isinstance(target_text, str) and target_text.strip():
                    reward_model["ground_truth"] = updated["target"]

            updated = attach_augmentation_metadata(
                updated,
                method_name=self.name,
                variant_idx=0,
                params={
                    "model": self._client.model,
                    "timeout": self._timeout,
                    "retry_attempts": self._retry_attempts,
                    "retry_delay": self._retry_delay,
                    "types": list(TYPES),
                },
                seed=rng_seed,
            )
            return [updated]
        except Exception as exc:
            logger.exception(
                "Skipping factual_anchor_typed sample after rewrite failure sample_id=%s, error=%s",
                sample_id, exc,
            )
            return []
