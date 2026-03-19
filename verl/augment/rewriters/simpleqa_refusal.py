"""Rewrite SimpleQA samples by replacing answers with random refusals."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
from typing import List

from ..registry import register
from ..schemas import attach_augmentation_metadata


@register("simpleqa_refusal")
class SimpleqaRefusalRewriter:
    name = "simpleqa_refusal"

    def __init__(self, *, refusal_answers_path: str | None = None) -> None:
        path = Path(refusal_answers_path) if refusal_answers_path else Path(__file__).resolve().parents[1] / "utils" / "refusal_answers.txt"
        self._refusal_answers_path = path
        self._refusal_answers = self._load_refusal_answers(path)

    def _load_refusal_answers(self, path: Path) -> list[str]:
        with path.open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    def rewrite(self, sample: dict, *, rng_seed: int | None = None) -> List[dict]:
        updated = deepcopy(sample)
        updated["answer"] = random.Random(rng_seed).choice(self._refusal_answers)
        updated["ability"] = "refusal"
        updated["reward_model"]["ground_truth"] = updated["answer"]
        updated = attach_augmentation_metadata(
            updated,
            method_name=self.name,
            variant_idx=0,
            params={"refusal_answers_path": str(self._refusal_answers_path)},
            seed=rng_seed,
        )
        return [updated]
