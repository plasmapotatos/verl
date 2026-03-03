from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None

from .datasets import get_dataset
from .io import coerce_id, coerce_responses, json_safe
from .metrics import compute_metrics
from .judge import OpenAIJudge

logger = logging.getLogger(__name__)


def run(
    dataset_name: str,
    input_parquet: str,
    output_json: str,
    use_judge: bool,
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    logger.info("Starting evaluation", extra={"dataset": dataset_name, "use_judge": use_judge})
    dataset_kwargs = {}
    if cache_dir:
        from pathlib import Path

        dataset_kwargs["cache_dir"] = Path(cache_dir)

    dataset = get_dataset(dataset_name, **dataset_kwargs)
    judge = OpenAIJudge() if use_judge else None

    df = pd.read_parquet(input_parquet)
    if limit is not None:
        df = df.head(limit)

    total_rows = len(df)
    logger.info("Loaded predictions", extra={"rows": total_rows})

    evaluations: List[str] = []
    rows: List[Dict[str, Any]] = []

    row_iter = df.iterrows()
    if tqdm is not None:
        row_iter = tqdm(row_iter, total=total_rows, desc="Evaluating", unit="row")

    for _, row in row_iter:
        record = row.to_dict()
        # print(record)
        sample_id = coerce_id(record)
        responses = coerce_responses(record)
        # print(f"Evaluating sample_id={sample_id} with responses={responses}")
        graders: List[Dict[str, Any]] = []

        for predicted in responses:
            try:
                sample = dataset.get_sample_by_id(sample_id)
                if use_judge and judge is not None:
                    prompt = dataset.build_judge_prompt(sample, predicted)
                    raw = judge.generate(prompt)
                    evaluation = dataset.parse_judge_output(raw)
                else:
                    prompt = ""
                    raw = ""
                    evaluation = dataset.rule_grade(sample, predicted)
            except Exception as exc:
                prompt = ""
                raw = str(exc)
                evaluation = "failed_to_parse"

            graders.append(
                {
                    "predicted_answer": predicted,
                    "prompt": prompt,
                    "raw_response": raw,
                    "evaluation": evaluation,
                }
            )
            evaluations.append(evaluation)

        record["coerced_id"] = sample_id
        record["coerced_responses"] = responses
        record["graders"] = graders
        rows.append(json_safe(record))

    metrics = compute_metrics(evaluations)
    output = {"metrics": json_safe(metrics), "rows": rows}

    import json

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("Saved evaluation output", extra={"output": output_json})

    return output
