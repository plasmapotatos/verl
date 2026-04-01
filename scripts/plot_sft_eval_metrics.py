#!/usr/bin/env python3
"""Plot SFT eval metrics with global-step x-axis."""

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("attempt_rate", "accuracy", "accuracy_attempted", "pass_at_k")


@dataclass(frozen=True)
class EvalPoint:
    lr: float
    epochs: int
    step: int
    dataset: str
    metrics: Dict[str, float]
    exp_name: str


def _parse_lr_ep(exp_name: str) -> Optional[Tuple[float, int]]:
    match = re.search(r"_lr([0-9eE.+-]+)_ep(?:max)?(\d+)", exp_name)
    if not match:
        return None
    return float(match.group(1)), int(match.group(2))


def _parse_step(text: str) -> int:
    match = re.search(r"global_step_(\d+)", text)
    if match:
        return int(match.group(1))
    if "/base/" in text or text.endswith("/base") or "/base/pass@k/" in text:
        return 0
    return -1


def _parse_dataset_label(filename: str) -> str:
    base = filename
    if base.endswith("_eval.json"):
        base = base[: -len("_eval.json")]
    if "__on_eval_" in base:
        return base.split("__on_eval_", 1)[1]
    if "__on_" in base:
        return base.split("__on_", 1)[1]
    return "unknown"


def _normalize_dataset_label(label: str) -> str:
    if label.startswith("eval_"):
        return label[len("eval_") :]
    return label


def _sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label)


def _dataset_from_eval_data(eval_data: str) -> str:
    name = Path(eval_data).name
    if name.endswith(".parquet"):
        name = name[: -len(".parquet")]
    return name


def _discover_experiment_dirs(project_dir: Path) -> List[Path]:
    if project_dir.name == "generations":
        return [project_dir.parent.parent]
    if project_dir.name.startswith("global_step_"):
        return [project_dir.parent]
    if list(project_dir.glob("global_step_*/generations")):
        return [project_dir]
    return sorted(
        p
        for p in project_dir.iterdir()
        if p.is_dir() and list(p.glob("global_step_*/generations"))
    )


def _iter_generation_dirs(exp_dir: Path) -> List[Path]:
    return sorted(p for p in exp_dir.glob("global_step_*/generations") if p.is_dir())


def _pass_k_summary_files(exp_dir: Path) -> List[Path]:
    patterns = (
        "global_step_*/pass@k/pass_at_k_*.json",
        "global_step_*/pass@k/*/pass_at_k.json",
        "base/pass@k/pass_at_k_*.json",
        "base/pass@k/*/pass_at_k.json",
    )
    files = set()
    for pattern in patterns:
        files.update(exp_dir.glob(pattern))
    return sorted(path for path in files if path.is_file())


def _collect_eval_points(project_dir: Path) -> List[EvalPoint]:
    points: List[EvalPoint] = []
    exp_dirs = _discover_experiment_dirs(project_dir)

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        lr_ep = _parse_lr_ep(exp_name)
        if exp_name == "base":
            lr, epochs = 0.0, 0
        elif lr_ep is not None:
            lr, epochs = lr_ep
        else:
            lr, epochs = 0.0, 0

        for gen_dir in _iter_generation_dirs(exp_dir):
            step = _parse_step(str(gen_dir.parent))
            for eval_file in sorted(gen_dir.glob("*_eval.json")):
                dataset_label = _normalize_dataset_label(_parse_dataset_label(eval_file.name))
                with eval_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
                if not isinstance(metrics, dict):
                    continue
                points.append(
                    EvalPoint(
                        lr=lr,
                        epochs=epochs,
                        step=step,
                        dataset=dataset_label,
                        metrics=metrics,
                        exp_name=exp_name,
                    )
                )

        for pass_file in _pass_k_summary_files(exp_dir):
            step = _parse_step(str(pass_file))
            with pass_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                continue

            eval_data = payload.get("eval_data")
            if not isinstance(eval_data, str):
                continue
            dataset_label = _dataset_from_eval_data(eval_data)
            dataset_label = _normalize_dataset_label(dataset_label)

            pass_info = payload.get("pass_at_k", {})
            pass_val = pass_info.get("pass_at_k") if isinstance(pass_info, dict) else None
            if pass_val is None:
                continue
            k = payload.get("k")
            metric_name = f"pass_at_k_{k}" if k is not None else "pass_at_k"
            metrics = {metric_name: float(pass_val)}
            if k is None:
                metrics["pass_at_k"] = float(pass_val)

            points.append(
                EvalPoint(
                    lr=lr,
                    epochs=epochs,
                    step=step,
                    dataset=dataset_label,
                    metrics=metrics,
                    exp_name=exp_name,
                )
            )

    return points


def _steps_per_epoch_label(points: List[EvalPoint]) -> Optional[str]:
    per_exp: Dict[str, float] = {}
    for point in points:
        if point.epochs <= 0 or point.step < 0:
            continue
        prev = per_exp.get(point.exp_name)
        per_exp[point.exp_name] = max(prev, float(point.step)) if prev is not None else float(point.step)

    ratios: List[float] = []
    for exp_name, max_step in per_exp.items():
        exp_epochs = max(p.epochs for p in points if p.exp_name == exp_name)
        if exp_epochs <= 0:
            continue
        ratios.append(max_step / float(exp_epochs))

    if not ratios:
        return None

    min_ratio = min(ratios)
    max_ratio = max(ratios)
    if max_ratio <= 0:
        return None
    if (max_ratio - min_ratio) <= max(1.0, 0.05 * max_ratio):
        return f"steps/epoch≈{int(round(sum(ratios) / len(ratios)))}"
    return f"steps/epoch≈{int(round(min_ratio))}-{int(round(max_ratio))}"


def _plot_metric(points: List[EvalPoint], output_dir: Path) -> None:
    by_dataset: Dict[str, List[EvalPoint]] = {}
    for point in points:
        by_dataset.setdefault(point.dataset, []).append(point)

    for dataset, dataset_points in by_dataset.items():
        dataset_dir = output_dir / _sanitize_label(dataset)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        metric_names = sorted({m for p in dataset_points for m in p.metrics.keys() if isinstance(p.metrics.get(m), (int, float))})
        for metric in metric_names:
            by_series: Dict[str, Dict[int, float]] = {}
            for point in dataset_points:
                if metric not in point.metrics:
                    continue
                x = point.step
                if x < 0:
                    continue
                series = point.exp_name
                by_series.setdefault(series, {})
                prev = by_series[series].get(x)
                val = float(point.metrics[metric])
                by_series[series][x] = max(prev, val) if prev is not None else val

            if not by_series:
                continue

            plt.figure(figsize=(7, 4.5))
            legend_handles = []
            legend_labels = []
            for series_name, xy in sorted(by_series.items(), key=lambda item: item[0]):
                xs = []
                ys = []
                for x in sorted(xy.keys()):
                    value = float(xy[x])
                    if not math.isfinite(value):
                        continue
                    xs.append(x)
                    ys.append(value)
                if not xs:
                    continue
                line, = plt.plot(xs, ys, marker="o", label=series_name)
                legend_handles.append(line)
                legend_labels.append(series_name)

            plt.title(f"{metric} vs step")
            plt.xlabel("global step")
            plt.ylabel(metric)
            plt.grid(True, linestyle="--", alpha=0.4)
            if legend_handles:
                plt.legend(legend_handles, legend_labels, loc="best", fontsize=9)
            steps_label = _steps_per_epoch_label(dataset_points)
            if steps_label:
                plt.gca().text(
                    0.98,
                    0.02,
                    steps_label,
                    transform=plt.gca().transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    alpha=0.7,
                )
            plt.tight_layout()

            out_path = dataset_dir / f"{_sanitize_label(metric)}_by_step.png"
            plt.savefig(out_path, dpi=150)
            plt.close()


def _plot_combined_metric_set(
    points: List[EvalPoint],
    combined_dir: Path,
    title: str,
    filename: str,
    metric_predicate: Callable[[str], bool],
    linestyle_fn: Callable[[str], str],
    label_formatter: Optional[Callable[[str], str]] = None,
) -> None:
    by_dataset_metric: Dict[Tuple[str, str], Dict[int, float]] = {}
    for point in points:
        x = point.step
        if x < 0:
            continue
        for metric, value in point.metrics.items():
            if not metric_predicate(metric):
                continue
            key = (point.dataset, metric)
            by_dataset_metric.setdefault(key, {})
            prev = by_dataset_metric[key].get(x)
            val = float(value)
            by_dataset_metric[key][x] = max(prev, val) if prev is not None else val

    if not by_dataset_metric:
        return

    plt.figure(figsize=(9, 6))
    legend_handles = []
    legend_labels = []
    for (dataset, metric), xy in sorted(by_dataset_metric.items(), key=lambda item: (item[0][0], item[0][1])):
        xs = []
        ys = []
        for x in sorted(xy.keys()):
            value = float(xy[x])
            if not math.isfinite(value):
                continue
            xs.append(x)
            ys.append(value)
        if not xs:
            continue
        linestyle = linestyle_fn(metric)
        label_metric = label_formatter(metric) if label_formatter else metric
        line, = plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.8,
            markersize=3.2,
            linestyle=linestyle,
            label=f"{dataset} | {label_metric}",
        )
        legend_handles.append(line)
        legend_labels.append(f"{dataset} | {label_metric}")

    plt.title(title)
    plt.xlabel("global step")
    plt.ylabel("score")
    plt.grid(True, linestyle="--", alpha=0.4)
    if legend_handles:
        plt.legend(legend_handles, legend_labels, loc="best", fontsize=8)
    steps_label = _steps_per_epoch_label(points)
    if steps_label:
        plt.gca().text(
            0.98,
            0.02,
            steps_label,
            transform=plt.gca().transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            alpha=0.7,
        )
    plt.tight_layout()

    out_path = combined_dir / filename
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_combined(points: List[EvalPoint], output_dir: Path) -> None:
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    _plot_combined_metric_set(
        points,
        combined_dir,
        title="accuracy + pass@k across datasets",
        filename="accuracy_and_passk_all_datasets_by_step.png",
        metric_predicate=lambda m: m == "accuracy" or m.startswith("pass_at_k"),
        linestyle_fn=lambda metric: "--" if metric.startswith("pass_at_k") else "-",
        label_formatter=lambda metric: metric.replace("pass_at_k", "pass@k"),
    )
    _plot_combined_metric_set(
        points,
        combined_dir,
        title="attempt rate across datasets",
        filename="attempt_rate_all_datasets_by_step.png",
        metric_predicate=lambda m: m == "attempt_rate",
        linestyle_fn=lambda _: "-.",
    )
    _plot_combined_metric_set(
        points,
        combined_dir,
        title="accuracy attempted across datasets",
        filename="accuracy_attempted_all_datasets_by_step.png",
        metric_predicate=lambda m: m == "accuracy_attempted",
        linestyle_fn=lambda _: "-",
        label_formatter=lambda metric: metric.replace("accuracy_attempted", "accuracy attempted"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir",
        required=True,
        help="Path to outputs/<project_name> directory or a single experiment dir",
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
    _plot_combined(points, output_dir)

    print(f"Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()
