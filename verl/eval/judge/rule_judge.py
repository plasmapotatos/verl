from __future__ import annotations

from .base import Judge


class RuleJudge(Judge):
    def generate(self, prompt: str) -> str:
        return ""
