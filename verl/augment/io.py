"""Parquet IO helpers for augmentation."""

from __future__ import annotations

from typing import Iterable, List

from datasets import Dataset


def read_parquet(path: str) -> Dataset:
    return Dataset.from_parquet(path)


def write_parquet(dataset_or_list: Dataset | Iterable[dict], path: str) -> None:
    if isinstance(dataset_or_list, Dataset):
        dataset = dataset_or_list
    else:
        dataset = Dataset.from_list(list(dataset_or_list))
    dataset.to_parquet(path)
