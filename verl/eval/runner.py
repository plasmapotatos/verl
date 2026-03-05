from __future__ import annotations

import concurrent.futures
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

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
    workers: Optional[int] = None,
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

    records = df.to_dict(orient="records")

    def _eval_record(index: int, record: Dict[str, Any]) -> Tuple[int, Dict[str, Any], List[str]]:
        sample_id = coerce_id(record)
        responses = coerce_responses(record)
        graders: List[Dict[str, Any]] = []
        evaluations_local: List[str] = []

        if use_judge:
            if not hasattr(_eval_record, "_thread_local"):
                _eval_record._thread_local = threading.local()
            thread_local = _eval_record._thread_local
            if getattr(thread_local, "judge", None) is None:
                thread_local.judge = OpenAIJudge()
            local_judge = thread_local.judge
        else:
            local_judge = None

        for predicted in responses:
            try:
                sample = dataset.get_sample_by_id(sample_id)
                if use_judge and local_judge is not None:
                    prompt = dataset.build_judge_prompt(sample, predicted)
                    raw = local_judge.generate(prompt)
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
            evaluations_local.append(evaluation)

        record["coerced_id"] = sample_id
        record["coerced_responses"] = responses
        record["graders"] = graders
        return index, json_safe(record), evaluations_local

    if workers is None or workers <= 1:
        row_iter = enumerate(records)
        if tqdm is not None:
            row_iter = tqdm(row_iter, total=total_rows, desc="Evaluating", unit="row")

        for idx, record in row_iter:
            _, row_out, evals_out = _eval_record(idx, record)
            rows.append(row_out)
            evaluations.extend(evals_out)
    else:
        rows = [None] * total_rows
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_eval_record, idx, record): idx for idx, record in enumerate(records)
            }
            if tqdm is not None:
                progress = tqdm(total=total_rows, desc="Evaluating", unit="row")
            else:
                progress = None

            for future in concurrent.futures.as_completed(futures):
                idx, row_out, evals_out = future.result()
                rows[idx] = row_out
                evaluations.extend(evals_out)
                if progress is not None:
                    progress.update(1)

            if progress is not None:
                progress.close()

        rows = [row for row in rows if row is not None]

    metrics = compute_metrics(evaluations)
    output = {"metrics": json_safe(metrics), "rows": rows}

    import json

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("Saved evaluation output", extra={"output": output_json})

    return output
