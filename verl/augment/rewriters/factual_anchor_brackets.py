"""Bracket factual anchors in prompt text using GPT-5 nano."""

from __future__ import annotations

from copy import deepcopy
import logging
import os
from pathlib import Path
from typing import List

from ..openai_client import OpenAIClient
from ..registry import register
from ..schemas import attach_augmentation_metadata, get_prompt_text, set_prompt_text


logger = logging.getLogger(__name__)

# File logger: writes one JSON line per failure to track which samples/fields fail
_FAIL_LOG_PATH = os.environ.get("FACTUAL_ANCHOR_FAIL_LOG", "tmp/factual_anchor_failures.jsonl")
_fail_handler: logging.FileHandler | None = None


def _get_fail_logger() -> logging.Logger:
    fail_logger = logging.getLogger("factual_anchor_failures")
    if not fail_logger.handlers:
        Path(_FAIL_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(_FAIL_LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        fail_logger.addHandler(handler)
        fail_logger.setLevel(logging.ERROR)
        fail_logger.propagate = False
    return fail_logger


def _load_system_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "factual_anchor_brackets.txt"
    return prompt_path.read_text(encoding="utf-8")


def _rewrite_text(
    client: OpenAIClient,
    system_prompt: str,
    text: str,
    *,
    seed: int | None,
    field: str = "unknown",
    sample_id: str | None = None,
) -> str:
    import json, time as _time
    # Strip control characters that can corrupt the JSON request body (e.g. null bytes).
    text = "".join(ch for ch in text if ch >= " " or ch in "\t\n\r")
    last_exc: Exception | None = None
    for attempt in range(client.retry_attempts):
        try:
            raw = client.generate(
                system_prompt=system_prompt,
                user_prompt=f"Text: {text}\nOutput:",
                seed=seed,
            )
            rewritten = raw.strip()
            if rewritten:
                return rewritten
            # Model returned empty — fall back to bracketing the whole text.
            logger.warning(f"Empty output for sample_id={sample_id} field={field}, falling back to whole-text bracket.")
            return f"[{text}]"
        except Exception as exc:
            last_exc = exc
        sleep_for = client.retry_delay * (2 ** attempt)
        if sleep_for > 0:
            _time.sleep(sleep_for)
    _get_fail_logger().error(json.dumps({
        "ts": _time.time(),
        "sample_id": sample_id,
        "field": field,
        "error": type(last_exc).__name__,
        "error_msg": str(last_exc),
        "text_len": len(text),
        "text_snippet": text[:200],
    }))
    raise last_exc  # type: ignore[misc]


@register("factual_anchor_brackets")
class FactualAnchorBracketsRewriter:
    name = "factual_anchor_brackets"

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
            rewritten_prompt = _rewrite_text(
                self._client, self._system_prompt, prompt_text, seed=rng_seed,
                field="prompt", sample_id=sample_id,
            )
            updated = set_prompt_text(updated, rewritten_prompt)

            question_text = sample.get("question")
            if isinstance(question_text, str) and question_text.strip():
                updated["question"] = _rewrite_text(
                    self._client,
                    self._system_prompt,
                    question_text,
                    seed=rng_seed,
                    field="question",
                    sample_id=sample_id,
                )

            answer_text = sample.get("answer")
            rewritten_answer = None
            if isinstance(answer_text, str) and answer_text.strip():
                rewritten_answer = _rewrite_text(
                    self._client, self._system_prompt, answer_text, seed=rng_seed,
                    field="answer", sample_id=sample_id,
                )
                updated["answer"] = rewritten_answer

            target_text = sample.get("target")
            if isinstance(target_text, str) and target_text.strip():
                rewritten_target = _rewrite_text(
                    self._client, self._system_prompt, target_text, seed=rng_seed,
                    field="target", sample_id=sample_id,
                )
                updated["target"] = rewritten_target

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
                },
                seed=rng_seed,
            )
            return [updated]
        except Exception as exc:
            logger.exception(f"Skipping factual_anchor_brackets sample after rewrite failure sample_id={sample_id}, error={exc}")
            return []