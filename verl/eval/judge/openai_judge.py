from __future__ import annotations

import os
from typing import Optional

from .base import Judge


class OpenAIJudge(Judge):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAIJudge.")

        ssl_cert_file = os.getenv("SSL_CERT_FILE")
        if ssl_cert_file and not os.path.exists(ssl_cert_file):
            os.environ.pop("SSL_CERT_FILE", None)

        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise ImportError(
                "openai package is required for OpenAIJudge. Install with `pip install openai`."
            ) from exc

        self._client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        message = response.choices[0].message
        return (message.content or "").strip()
