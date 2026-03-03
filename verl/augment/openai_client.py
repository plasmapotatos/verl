"""Minimal OpenAI client wrapper for augmentation utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


class OpenAIClient:
    def __init__(self, model: str) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please export OPENAI_API_KEY=<your_key>."
            )
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise RuntimeError("OpenAI SDK is required. Install the 'openai' package.") from exc

        try:
            import httpx  # type: ignore
        except Exception as exc:
            raise RuntimeError("httpx is required by the OpenAI SDK.") from exc

        # Avoid inheriting invalid SSL_CERT_FILE from the environment.
        http_client = httpx.Client(trust_env=False)
        self._client = OpenAI(api_key=api_key, http_client=http_client)
        self.model = model

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
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
        return outputs[0]
