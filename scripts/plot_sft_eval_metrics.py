#!/usr/bin/env python3
"""Plot SFT eval metrics by learning rate and epoch."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("attempt_rate", "accuracy", "accuracy_attempted")


@dataclass(frozen=True)
class EvalPoint:
    lr: float
    epochs: int
    step: int
    dataset: str
    metrics: Dict[str, float]


def _parse_lr_ep(exp_name: str) -> Tuple[float, int] | None:
    match = re.search(r"_lr([0-9eE.+-]+)_ep(\d+)", exp_name)
    if not match:
        return None
    lr = float(match.group(1))
    epochs = int(match.group(2))
    return lr, epochs


def _format_lr(lr: float) -> str:
    return f"{lr:.1e}"


def _parse_step(filename: str) -> int:
    match = re.search(r"global_step_(\d+)", filename)
    if not match:
        return -1
    return int(match.group(1))


def _parse_dataset_label(filename: str) -> str:
    base = filename
    if base.endswith("_eval.json"):
        base = base[: -len("_eval.json")]
    if "__on_" not in base:
        return "unknown"
    return base.split("__on_", 1)[1]


def _sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label)


def _collect_eval_points(project_dir: Path) -> List[EvalPoint]:
    points: List[EvalPoint] = []
    eval_files = list(project_dir.glob("*/generations/*_eval.json"))
    best_by_exp_dataset: Dict[Tuple[str, str], EvalPoint] = {}
    baseline_by_dataset: Dict[str, EvalPoint] = {}

    for eval_file in eval_files:
        exp_name = eval_file.parent.parent.name
        if exp_name == "base":
            lr = 0.0
            epochs = 0
        else:
            lr_ep = _parse_lr_ep(exp_name)
            if lr_ep is None:
                continue
            lr, epochs = lr_ep
        dataset_label = _parse_dataset_label(eval_file.name)
        step = _parse_step(eval_file.name)

        with eval_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}

        point = EvalPoint(
            lr=lr,
            epochs=epochs,
            step=step,
            dataset=dataset_label,
            metrics=metrics,
        )
        if exp_name == "base":
            if dataset_label not in baseline_by_dataset or step > baseline_by_dataset[dataset_label].step:
                baseline_by_dataset[dataset_label] = point
        else:
            key = (exp_name, dataset_label)
            if key not in best_by_exp_dataset or step > best_by_exp_dataset[key].step:
                best_by_exp_dataset[key] = point

    points.extend(best_by_exp_dataset.values())
    points.extend(baseline_by_dataset.values())
    return points


def _plot_metric(points: List[EvalPoint], output_dir: Path) -> None:
    by_dataset: Dict[str, List[EvalPoint]] = {}
    for point in points:
        by_dataset.setdefault(point.dataset, []).append(point)

    for dataset, dataset_points in by_dataset.items():
        dataset_label = _sanitize_label(dataset)
        dataset_dir = output_dir / dataset_label
        dataset_dir.mkdir(parents=True, exist_ok=True)

        by_lr: Dict[float, List[EvalPoint]] = {}
        baseline_point = None
        for point in dataset_points:
            if point.epochs == 0:
                baseline_point = point
            else:
                by_lr.setdefault(point.lr, []).append(point)

        for metric in METRICS:
            metric_dir = dataset_dir / metric
            metric_dir.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(7, 4.5))
            for lr, lr_points in sorted(by_lr.items(), key=lambda item: item[0]):
                lr_points = sorted(lr_points, key=lambda p: p.epochs)
                epochs = [p.epochs for p in lr_points]
                values = [p.metrics.get(metric, 0.0) for p in lr_points]
                if baseline_point is not None:
                    epochs = [0] + epochs
                    values = [baseline_point.metrics.get(metric, 0.0)] + values
                plt.plot(epochs, values, marker="o", label=f"lr={_format_lr(lr)}")

            plt.title(f"{metric} vs epochs")
            plt.xlabel("epochs")
            plt.ylabel(metric)
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend(loc="best", fontsize=9)
            plt.tight_layout()

            out_path = metric_dir / "all_lrs.png"
            plt.savefig(out_path, dpi=150)
            plt.close()


def _plot_metric_all_datasets(points: List[EvalPoint], output_dir: Path) -> None:
    by_dataset: Dict[str, Dict[int, List[EvalPoint]]] = {}
    baseline_by_dataset: Dict[str, EvalPoint] = {}
    for point in points:
        if point.epochs == 0:
            baseline_by_dataset[point.dataset] = point
            continue
        by_dataset.setdefault(point.dataset, {}).setdefault(point.epochs, []).append(point)

    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        plt.figure(figsize=(7, 4.5))
        for dataset, epochs_map in sorted(by_dataset.items(), key=lambda item: item[0]):
            epochs = sorted(epochs_map.keys())
            values = []
            for ep in epochs:
                ep_points = epochs_map[ep]
                values.append(max(p.metrics.get(metric, 0.0) for p in ep_points))

            baseline = baseline_by_dataset.get(dataset)
            if baseline is not None:
                epochs = [0] + epochs
                values = [baseline.metrics.get(metric, 0.0)] + values

            plt.plot(epochs, values, marker="o", label=dataset)

        plt.title(f"{metric} vs epochs (best across LRs)")
        plt.xlabel("epochs")
        plt.ylabel(metric)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(loc="best", fontsize=9)
        plt.tight_layout()

        out_path = combined_dir / f"{metric}_all_datasets.png"
        plt.savefig(out_path, dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir",
        required=True,
        help="Path to outputs/<project_name> directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write plots",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    points = _collect_eval_points(project_dir)
    if not points:
        raise RuntimeError("No eval JSON files found under project directory")

    os.makedirs(output_dir, exist_ok=True)
    _plot_metric(points, output_dir)
    _plot_metric_all_datasets(points, output_dir)

    print(f"Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()
