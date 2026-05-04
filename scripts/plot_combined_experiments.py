#!/usr/bin/env python3
"""Combine plot_sft_eval_metrics plots across multiple experiment dirs.

Reuses helpers from plot_sft_eval_metrics.py so the single-experiment script
stays the source of truth for collection + plot styling. Each experiment's
points are lightly retagged so:

  * per-dataset plots keep one subdir per dataset and add the experiment
    label to the series name (exp_name);
  * the "combined across datasets" plots treat each experiment's datasets
    as separate virtual datasets, which distinguishes them in the legend
    without modifying _plot_combined_metric_set.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_sft_eval_metrics import (  # noqa: E402
    EvalPoint,
    _collect_eval_points,
    _plot_combined,
    _plot_metric,
)


def _default_tag_for_path(path: Path) -> str:
    parts = path.parts
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return path.name


def _parse_spec(spec: str) -> Tuple[str, Path]:
    if "=" in spec:
        tag, raw_path = spec.split("=", 1)
        return tag, Path(raw_path).expanduser().resolve()
    path = Path(spec).expanduser().resolve()
    return _default_tag_for_path(path), path


def _uniquify_tags(specs: List[Tuple[str, Path]]) -> List[Tuple[str, Path]]:
    seen: dict = {}
    out: List[Tuple[str, Path]] = []
    for tag, path in specs:
        if tag in seen:
            seen[tag] += 1
            tag = f"{tag}#{seen[tag]}"
        else:
            seen[tag] = 1
        out.append((tag, path))
    return out


def _retag_for_per_dataset(points: List[EvalPoint], tag: str) -> List[EvalPoint]:
    return [replace(p, exp_name=f"{tag} | {p.exp_name}") for p in points]


def _retag_for_combined(points: List[EvalPoint], tag: str) -> List[EvalPoint]:
    return [replace(p, dataset=f"{tag} | {p.dataset}") for p in points]


def _default_output_dir(specs: List[Tuple[str, Path]]) -> Path:
    repo_root = _SCRIPT_DIR.parent
    joined = "__".join(
        tag.replace("/", "_").replace(" ", "_") for tag, _ in specs
    )
    return repo_root / "outputs" / "combined_plots" / joined


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine plot_sft_eval_metrics plots across multiple experiment "
            "project/experiment directories. Every plot the single-experiment "
            "script produces is reproduced with all experiments overlaid."
        ),
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        help=(
            "Experiment project dirs or experiment dirs (anything "
            "plot_sft_eval_metrics accepts as --project-dir). Optionally "
            "prefix with 'tag=' to set the legend label, e.g. "
            "'unbracketed=/path/to/exp'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write combined plots. Defaults to "
            "<repo>/outputs/combined_plots/<joined-tags>."
        ),
    )
    args = parser.parse_args()

    specs = _uniquify_tags([_parse_spec(s) for s in args.experiments])

    per_dataset_points: List[EvalPoint] = []
    combined_points: List[EvalPoint] = []
    for tag, project_dir in specs:
        if not project_dir.exists():
            raise FileNotFoundError(f"Project directory not found: {project_dir}")
        points = _collect_eval_points(project_dir)
        if not points:
            print(
                f"warning: no eval JSON files found under {project_dir}",
                file=sys.stderr,
            )
            continue
        per_dataset_points.extend(_retag_for_per_dataset(points, tag))
        combined_points.extend(_retag_for_combined(points, tag))

    if not per_dataset_points:
        raise RuntimeError("No eval JSON files found under any project directory")

    if args.output_dir is None:
        output_dir = _default_output_dir(specs)
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_metric(per_dataset_points, output_dir)
    _plot_combined(combined_points, output_dir)

    print(f"Wrote combined plots to: {output_dir}")


if __name__ == "__main__":
    main()
