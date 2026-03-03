"""LLM prompt-based rewriter with mode-specific system prompts."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

from ..registry import register
from ..schemas import get_prompt_text


def _load_system_prompt(mode: str) -> str:
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / f"{mode}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Prompt template not found for mode '{mode}'. Expected file: {prompt_path}"
    )


def _extract_question_segment(prompt_text: str) -> Optional[tuple[str, str, str]]:
    """Return (prefix, question, suffix) if a Question: line exists."""
    match = re.search(r"(Question:\s*)([^\n]*)(.*)", prompt_text, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def _replace_question_segment(prompt_text: str, new_question: str) -> str:
    extracted = _extract_question_segment(prompt_text)
    if extracted is None:
        return new_question
    prefix, _old, suffix = extracted
    return f"{prefix}{new_question}{suffix}"


class OpenAIClient:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please export OPENAI_API_KEY=<your_key>."
            )
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "OpenAI SDK is required. Install the 'openai' package."
            ) from exc
        ssl_cert_file = os.getenv("SSL_CERT_FILE")
        if ssl_cert_file and not Path(ssl_cert_file).exists():
            try:
                import httpx  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "SSL_CERT_FILE is set but invalid, and httpx is unavailable to override. "
                    "Unset SSL_CERT_FILE or point it to a valid CA bundle."
                ) from exc
            http_client = httpx.Client(trust_env=False)
            self._client = OpenAI(api_key=api_key, http_client=http_client)
        else:
            self._client = OpenAI(api_key=api_key)

    def rewrite(
        self,
        text: str,
        *,
        n: int,
        temperature: float,
        style: str | None,
        system_prompt: str,
        seed: int | None,
    ) -> List[str]:
        user_prompt = "Rewrite the following text."
        if style:
            user_prompt += f" Style: {style}."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt}\n\n{text}"},
        ]
        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=temperature,
                n=n,
                seed=seed,
            )
        except Exception as exc:  # pragma: no cover - exercised in live usage
            raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

        outputs: List[str] = []
        for choice in response.choices:
            content = choice.message.content
            if content is None:
                continue
            outputs.append(content.strip())
        if not outputs:
            raise RuntimeError("OpenAI API returned no content.")
        return outputs


@register("llm_rewriter")
@register("llm_paraphrase")
class LlmRewriter:
    name = "llm_rewriter"

    def __init__(
        self,
        *,
        mode: str = "paraphrase",
        n: int = 1,
        temperature: float = 0.7,
        style: str | None = None,
        preserve_format: bool = True,
    ) -> None:
        self.mode = mode
        self.n = n
        self.temperature = temperature
        self.style = style
        self.preserve_format = preserve_format
        self._client = OpenAIClient()
        self._system_prompt = _load_system_prompt(mode)

    def rewrite(self, sample: dict, *, rng_seed: int | None = None) -> List[dict]:
        prompt_text = get_prompt_text(sample)

        if self.preserve_format:
            extracted = _extract_question_segment(prompt_text)
            if extracted is not None:
                _prefix, question, _suffix = extracted
                text_to_rewrite = question
            else:
                text_to_rewrite = prompt_text
        else:
            text_to_rewrite = prompt_text

        rewrites = self._client.rewrite(
            text_to_rewrite,
            n=self.n,
            temperature=self.temperature,
            style=self.style,
            system_prompt=self._system_prompt,
            seed=rng_seed,
        )

        outputs: List[dict] = []
        for idx, variant in enumerate(rewrites):
            updated = deepcopy(sample)
            if self.preserve_format:
                updated_text = _replace_question_segment(prompt_text, variant)
            else:
                updated_text = variant
            updated["prompt"][0]["content"] = updated_text

            extra_info = updated.get("extra_info")
            if extra_info is None or not isinstance(extra_info, dict):
                extra_info = {}
                updated["extra_info"] = extra_info
            extra_info["augmentation"] = {
                "method": self.name,
                "mode": self.mode,
                "variant_index": idx,
                "params": {
                    "temperature": self.temperature,
                    "style": self.style,
                    "preserve_format": self.preserve_format,
                },
            }
            outputs.append(updated)
        return outputs
