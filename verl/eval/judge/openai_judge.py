from __future__ import annotations

import os
from typing import Optional

from .base import Judge


class OpenAIJudge(Judge):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5-nano",
        max_prompt_tokens: int = 16_000,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_prompt_tokens = max_prompt_tokens
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

    def _truncate_prompt(self, prompt: str) -> str:
        if self.max_prompt_tokens <= 0:
            return prompt

        try:
            import tiktoken  # type: ignore

            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")

            tokens = encoding.encode(prompt)
            if len(tokens) <= self.max_prompt_tokens:
                return prompt
            return encoding.decode(tokens[: self.max_prompt_tokens])
        except Exception:
            max_chars = self.max_prompt_tokens * 4
            if len(prompt) <= max_chars:
                return prompt
            return prompt[:max_chars]

    def generate(self, prompt: str) -> str:
        prompt = self._truncate_prompt(prompt)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            # temperature=0,
        )
        message = response.choices[0].message
        return (message.content or "").strip()
