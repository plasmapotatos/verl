"""Registry for augmentation rewriters."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List

from .rewriters.base import Rewriter

_REGISTRY: Dict[str, Callable[[], Rewriter]] = {}


def register(name: str) -> Callable[[Callable[[], Rewriter]], Callable[[], Rewriter]]:
    """Register a rewriter factory under a method name."""

    def decorator(factory: Callable[[], Rewriter]) -> Callable[[], Rewriter]:
        _REGISTRY[name] = factory
        return factory

    return decorator


def _filter_kwargs(factory: Callable[..., Rewriter], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return {}

    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs

    allowed = {
        key
        for key, param in sig.parameters.items()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    return {key: value for key, value in kwargs.items() if key in allowed}


def get(name: str, **kwargs: Any) -> Rewriter:
    _ensure_default_rewriters_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown rewriter method: {name}")
    factory = _REGISTRY[name]
    filtered_kwargs = _filter_kwargs(factory, kwargs)
    return factory(**filtered_kwargs)


def list_methods() -> List[str]:
    _ensure_default_rewriters_loaded()
    return sorted(_REGISTRY.keys())


def _ensure_default_rewriters_loaded() -> None:
    """Import rewriters to populate the registry (idempotent)."""
    if _REGISTRY:
        return
    # noqa: F401 - import for side effects
    from .rewriters import llm_rewriter, simpleqa_web_search, templates  # type: ignore
