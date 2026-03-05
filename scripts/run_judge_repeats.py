#!/usr/bin/env python3
"""Run VERL judge evaluation multiple times and aggregate metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

from verl.eval.runner import run as run_eval


def _default_output_dir(input_path: str, runs: int) -> Path:
    base = Path(input_path)
    name = f"{base.stem}_judge_runs_gpt4o-mini_{runs}"
    return base.with_name(name)


def _collect_numeric_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    numeric: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            numeric[key] = float(value)
    return numeric


def _summarize_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not all_metrics:
        return {}

    keys = sorted({key for metrics in all_metrics for key in metrics.keys()})
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = [metrics.get(key, 0.0) for metrics in all_metrics]
        summary[key] = {
            "mean": mean(values),
            "std": pstdev(values),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input predictions parquet")
    parser.add_argument("--dataset", required=True, help="Dataset name for eval")
    parser.add_argument("--runs", type=int, required=True, help="Number of judge runs")
    parser.add_argument("--output-dir", default=None, help="Directory to write outputs")
    parser.add_argument("--cache-dir", default=None, help="Dataset cache directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows")
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be a positive integer")

    input_path = args.input
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(input_path, args.runs)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_run_metrics: List[Dict[str, float]] = []
    output_paths: List[str] = []

    for run_idx in range(1, args.runs + 1):
        output_path = output_dir / f"judge_run_{run_idx}.json"
        result = run_eval(
            dataset_name=args.dataset,
            input_parquet=input_path,
            output_json=str(output_path),
            use_judge=True,
            cache_dir=args.cache_dir,
            limit=args.limit,
        )
        metrics = _collect_numeric_metrics(result.get("metrics", {}))
        per_run_metrics.append(metrics)
        output_paths.append(str(output_path))
        print(f"Wrote: {output_path}")

    summary = {
        "input": input_path,
        "dataset": args.dataset,
        "runs": args.runs,
        "outputs": output_paths,
        "metrics": _summarize_metrics(per_run_metrics),
        "per_run_metrics": per_run_metrics,
    }

    summary_path = output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
