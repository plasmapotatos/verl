"""A simple template-based rewriter for prompt augmentation."""

from __future__ import annotations

from typing import Any, Dict, List

from ..registry import register
from ..schemas import attach_augmentation_metadata, get_prompt_text, set_prompt_text


@register("templates")
class TemplatesRewriter:
    name = "templates"

    def rewrite(self, sample: dict, *, rng_seed: int | None = None) -> List[dict]:
        original = get_prompt_text(sample)
        rewritten = f"Reworded: {original}"
        updated = set_prompt_text(sample, rewritten)
        updated = attach_augmentation_metadata(
            updated,
            method_name=self.name,
            variant_idx=0,
            params={"prefix": "Reworded: "},
            seed=rng_seed,
        )
        return [updated]
