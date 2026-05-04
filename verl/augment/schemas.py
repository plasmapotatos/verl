"""Schema helpers for augmentation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Set


ALLOWED_MUTABLE_FIELDS: Set[str] = {"prompt", "extra_info"}


def get_mutable_fields() -> Set[str]:
    """Return top-level mutable fields allowed for augmentation."""
    return set(ALLOWED_MUTABLE_FIELDS)


def get_prompt_text(sample: Dict[str, Any]) -> str:
    prompt = sample.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("sample['prompt'] must be a non-empty list")
    first = prompt[0]
    if not isinstance(first, dict) or "content" not in first:
        raise ValueError("prompt[0] must be a dict with a 'content' field")
    return str(first["content"])


def set_prompt_text(sample: Dict[str, Any], new_text: str) -> Dict[str, Any]:
    updated = deepcopy(sample)
    prompt = updated.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("sample['prompt'] must be a non-empty list")
    if not isinstance(prompt[0], dict):
        raise ValueError("prompt[0] must be a dict")
    prompt[0]["content"] = new_text
    return updated


def attach_augmentation_metadata(
    sample: Dict[str, Any],
    method_name: str,
    variant_idx: int,
    params: Dict[str, Any] | None,
    seed: int | None,
) -> Dict[str, Any]:
    updated = deepcopy(sample)
    extra_info = updated.get("extra_info")
    if extra_info is None or not isinstance(extra_info, dict):
        extra_info = {}
        updated["extra_info"] = extra_info
    original_sample_id = extra_info.get("sample_id")
    # pyarrow cannot serialize an empty struct; ensure params has at least
    # one field so Parquet writes succeed even when the caller passes {}.
    safe_params = dict(params) if params else {}
    if not safe_params:
        safe_params = {"_": ""}
    extra_info["augmentation"] = {
        "method": method_name,
        "variant_index": variant_idx,
        "params": safe_params,
        "seed": seed,
        "original_sample_id": original_sample_id,
    }
    return updated
