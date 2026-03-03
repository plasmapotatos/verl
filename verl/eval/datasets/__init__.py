"""Dataset adapters for VERL evaluation."""

from .registry import get_dataset, list_datasets, register_dataset
from .simpleqa import SimpleQADataset

__all__ = ["get_dataset", "list_datasets", "register_dataset", "SimpleQADataset"]
