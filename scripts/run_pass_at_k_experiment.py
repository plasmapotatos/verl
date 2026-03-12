#!/usr/bin/env python3
"""Run pass@k over all merged checkpoints inside an experiment.

python scripts/run_pass_at_k_experiment.py \
  --experiment-dir /work/hdd/bbsg/twei2/rl/verl/outputs/rl/simpleqa_rich_sft_grpo_train_eval/simpleqa_rich_sft_grpo_train_eval \
  --eval-data /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/rich_sft/simpleqa_rich_sft_train_origqa_frac0.9_train_frac0.1.parquet \
  --ks 32

"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_checkpoints(experiment_dir: Path) -> Sequence[Path]:
    return sorted(
        ckpt
        for ckpt in experiment_dir.iterdir()
        if ckpt.is_dir() and ckpt.name.startswith("global_step_") and (ckpt / "merged_hf_model").is_dir()
    )


def _parse_ks(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_pass_at_k(
    pass_script: Path,
    checkpoint_path: Path,
    dataset: str,
    eval_data: Path,
    output_dir: Path,
    ks: Iterable[int],
    extra_args: Sequence[str] | None,
) -> None:
    for k in ks:
        print(f"Running pass@{k} for checkpoint {checkpoint_path.name}")
        cmd = [
            sys.executable,
            str(pass_script),
            "--checkpoint",
            str(checkpoint_path),
            "--dataset",
            dataset,
            "--eval-data",
            str(eval_data),
            "--output-dir",
            str(output_dir),
            "--top-k",
            str(k),
        ]
        if extra_args:
            cmd.extend(extra_args)
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pass@k for every merged checkpoint")
    parser.add_argument("--experiment-dir", required=True, type=Path, help="Experiment directory containing global_step_* checkpoints")
    parser.add_argument("--dataset", default="simpleqa", help="Dataset name for evaluation")
    parser.add_argument("--eval-data", required=True, type=Path, help="Path to eval parquet file")
    parser.add_argument(
        "--ks",
        required=True,
        help="Comma-separated list of k values (e.g. 1,2,4)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Base directory for pass@k outputs (defaults to <experiment>/pass_at_k)",
    )
    parser.add_argument(
        "--pass-at-k-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to scripts/pass_at_k.py",
    )
    args = parser.parse_args()

    if not args.experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {args.experiment_dir}")

    checkpoints = _find_checkpoints(args.experiment_dir)
    if not checkpoints:
        print("No merged checkpoints found under", args.experiment_dir)
        return

    ks = _parse_ks(args.ks)
    if not ks:
        raise ValueError("At least one k value must be provided")

    base_output = args.output_dir or args.experiment_dir / "pass_at_k"
    base_output.mkdir(parents=True, exist_ok=True)

    script_path = _repo_root() / "scripts" / "pass_at_k.py"
    if not script_path.exists():
        raise FileNotFoundError(f"pass_at_k.py not found at expected location: {script_path}")

    extra_args = args.pass_at_k_args

    for ckpt in checkpoints:
        merged = ckpt / "merged_hf_model"
        per_ckpt_output = base_output / ckpt.name
        per_ckpt_output.mkdir(parents=True, exist_ok=True)
        run_pass_at_k(
            pass_script=script_path,
            checkpoint_path=merged,
            dataset=args.dataset,
            eval_data=args.eval_data,
            output_dir=per_ckpt_output,
            ks=ks,
            extra_args=extra_args,
        )


if __name__ == "__main__":
    main()
