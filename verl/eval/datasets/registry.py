from __future__ import annotations

from typing import Any, Callable, Dict, Type

from .base import DatasetAdapter

_DATASET_REGISTRY: Dict[str, Type[DatasetAdapter]] = {}


def register_dataset(name: str) -> Callable[[Type[DatasetAdapter]], Type[DatasetAdapter]]:
    def decorator(cls: Type[DatasetAdapter]) -> Type[DatasetAdapter]:
        _DATASET_REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


def get_dataset(name: str, **kwargs: Any) -> DatasetAdapter:
    if name not in _DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}. Available: {list_datasets()}")
    return _DATASET_REGISTRY[name](**kwargs)


def list_datasets() -> list[str]:
    return sorted(_DATASET_REGISTRY.keys())
