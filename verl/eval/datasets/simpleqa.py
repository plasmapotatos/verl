from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve

import pandas as pd

from .base import DatasetAdapter
from .registry import register_dataset
from verl.utils.reward_score.simpleqa import compute_score as _compute_score

DEFAULT_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
DEFAULT_CACHE_DIR = Path(os.path.expanduser("./data/simpleqa"))


def _default_cache_path(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(DEFAULT_URL).path).name or "simpleqa.csv"
    return cache_dir / filename


def _load_prompt_template() -> str:
    prompt_path = Path(__file__).resolve().parents[1] / "judge" / "prompts" / "simpleqa_judge.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt template not found at {prompt_path}. Please ensure the file exists.")


@register_dataset("simpleqa")
class SimpleQADataset(DatasetAdapter):
    def __init__(
        self,
        url: str = DEFAULT_URL,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.url = url
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_path = _default_cache_path(self.cache_dir)
        self._samples: Dict[str, Dict] = {}
        self._prompt_template = _load_prompt_template()
        self._ensure_dataset()

    def _ensure_dataset(self) -> None:
        if not self.cache_path.exists():
            urlretrieve(self.url, self.cache_path)
        print(f"Loading dataset from {self.cache_path}")
        df = pd.read_csv(self.cache_path)
        for _, row in df.iterrows():
            sample_id = str(row.get("problem_id", len(self._samples)))
            sample = {
                "id": sample_id,
                "question": row.get("problem", ""),
                "answer": row.get("answer", ""),
                "metadata": self._parse_metadata(row.get("metadata")),
            }
            if sample_id:
                self._samples[sample_id] = sample

    @staticmethod
    def _parse_metadata(value) -> Dict:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except Exception:
                return {}
        return {}

    def get_sample_by_id(self, sample_id: str) -> Dict:
        if sample_id not in self._samples:
            raise KeyError(f"Sample id not found: {sample_id}")
        return self._samples[sample_id]

    def build_judge_prompt(self, sample: Dict, predicted_answer: str) -> str:
        template = self._prompt_template
        if not all(tag in template for tag in ("{question}", "{target}", "{predicted_answer}")):
            raise ValueError("Prompt template is missing required placeholders: {question}, {target}, {predicted_answer}")
        return template.format(
            question=sample.get("question", ""),
            target=sample.get("answer", ""),
            predicted_answer=predicted_answer,
        )

    def parse_judge_output(self, judge_output: str) -> str:
        text = (judge_output or "").strip().upper()
        if "NOT_ATTEMPTED" in text:
            return "not_attempted"
        if "CORRECT" in text or " A " in f" {text} ":
            return "correct"
        if " C " in f" {text} ":
            return "not_attempted"
        return "incorrect"

    def rule_grade(self, sample: Dict, predicted_answer: str) -> str:
        ground_truth = str(sample.get("answer", ""))
        score = _compute_score(predicted_answer, ground_truth, reward_mode="binary")
        if score == 1.0:
            return "correct"
        if score == 0.0:
            return "not_attempted"
        return "incorrect"
