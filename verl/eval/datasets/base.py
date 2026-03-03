from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class DatasetAdapter(ABC):
    name: str

    @abstractmethod
    def get_sample_by_id(self, sample_id: str) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def build_judge_prompt(self, sample: Dict, predicted_answer: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def parse_judge_output(self, judge_output: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def rule_grade(self, sample: Dict, predicted_answer: str) -> str:
        raise NotImplementedError
