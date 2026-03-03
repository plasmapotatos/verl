"""Base protocol for augmentation rewriters."""

from __future__ import annotations

from typing import Protocol


class Rewriter(Protocol):
    name: str

    def rewrite(self, sample: dict, *, rng_seed: int | None = None) -> list[dict]:
        """Return 1..K augmented samples for a given input sample."""
        ...
