#!/usr/bin/env python3
"""Run pass@k over all merged checkpoints inside an experiment.

python scripts/run_pass_at_k_experiment.py \
  --experiment-dir /work/hdd/bbsg/twei2/rl/verl/outputs/rl/simpleqa_rich_sft_grpo_train_eval/simpleqa_rich_sft_grpo_train_eval \
  --eval-data /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/rich_sft/simpleqa_rich_sft_train_origqa_frac0.9_train_frac0.1.parquet \
  --ks 32

"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


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


def _sanitize_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    return payload if isinstance(payload, dict) else None


def _result_exists(output_dir: Path, eval_stem: str, eval_data: Path, k: int, multi_k: bool, layout: str) -> bool:
    pass_k_dir = output_dir / "pass@k"

    if layout == "dataset_subdir":
        summary_path = pass_k_dir / eval_stem / "pass_at_k.json"
        payload = _load_json(summary_path)
        if not payload:
            return False
        return payload.get("eval_data") == str(eval_data) and payload.get("k") == k

    suffix = f"_k{k}" if multi_k else ""
    summary_path = pass_k_dir / f"pass_at_k_eval_{eval_stem}{suffix}.json"
    payload = _load_json(summary_path)
    if not payload:
        return False
    return payload.get("eval_data") == str(eval_data) and payload.get("k") == k


def _finalize_outputs(output_dir: Path, eval_stem: str, k: int, multi_k: bool, layout: str) -> None:
    pass_k_dir = output_dir / "pass@k"
    if not pass_k_dir.is_dir():
        return

    if layout == "dataset_subdir":
        target_dir = pass_k_dir / eval_stem
        target_dir.mkdir(parents=True, exist_ok=True)

        file_map = {
            pass_k_dir / f"generations_{k}.parquet": target_dir / "generations.parquet",
            pass_k_dir / f"eval_{k}.json": target_dir / "eval.json",
            pass_k_dir / f"pass_at_k_{k}.json": target_dir / "pass_at_k.json",
        }
        for src, dst in file_map.items():
            if src.exists():
                shutil.move(str(src), str(dst))
        return

    src = pass_k_dir / f"pass_at_k_{k}.json"
    if not src.exists():
        return
    suffix = f"_k{k}" if multi_k else ""
    dst = pass_k_dir / f"pass_at_k_eval_{eval_stem}{suffix}.json"
    src.replace(dst)


def run_pass_at_k(
    pass_script: Path,
    checkpoint_path: Path,
    dataset: str,
    eval_data: Path,
    output_dir: Path,
    ks: Iterable[int],
    extra_args: Optional[Sequence[str]],
    layout: str,
) -> None:
    eval_stem = _sanitize_label(eval_data.stem)
    ks_list = list(ks)
    multi_k = len(ks_list) > 1

    for k in ks_list:
        if _result_exists(output_dir, eval_stem, eval_data, k, multi_k, layout):
            print(f"Skipping pass@{k} for checkpoint {checkpoint_path.name} (already exists)")
            continue
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
        _finalize_outputs(output_dir, eval_stem, k, multi_k, layout)


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
        help="Base directory containing checkpoint subdirs (defaults to <experiment>)",
    )
    parser.add_argument(
        "--pass-at-k-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to scripts/pass_at_k.py",
    )
    parser.add_argument(
        "--layout",
        choices=("rename_summary", "dataset_subdir"),
        default="rename_summary",
        help="How to organize outputs under pass@k/",
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

    base_output = args.output_dir or args.experiment_dir
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
            layout=args.layout,
        )


if __name__ == "__main__":
    main()
