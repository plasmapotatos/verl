#!/usr/bin/env python3
"""Probe for catastrophic forgetting by evaluating a checkpoint on a baseline
dataset (things the model originally knew).

Last RL checkpoint:
    python scripts/probe_forgetting.py \
        --experiment-dir outputs/rl/simpleqa_factual_anchor_grpo_smoketest/binary

Base model (infer from experiment's base/pass@k/ summaries):
    python scripts/probe_forgetting.py \
        --experiment-dir outputs/rl/simpleqa_factual_anchor_grpo_smoketest/binary \
        --base

Base model (explicit, no experiment dir needed):
    python scripts/probe_forgetting.py --base --model Qwen/Qwen2.5-3B-Instruct

Any model standalone:
    python scripts/probe_forgetting.py --model /path/to/checkpoint
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_EVAL_DATA = _REPO_ROOT / "data" / "simpleqa" / "partition" / "knows.parquet"
_DEFAULT_KS = [1, 16, 32]
_DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def _find_last_checkpoint(experiment_dir: Path) -> Path:
    candidates = sorted(
        (ckpt for ckpt in experiment_dir.iterdir() if ckpt.is_dir() and ckpt.name.startswith("global_step_")),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No global_step_* checkpoints found in {experiment_dir}")
    return candidates[-1]


def _infer_base_model(experiment_dir: Path) -> str:
    """Read the base model path from an existing pass@k summary in <exp>/base/pass@k/."""
    base_pass_k = experiment_dir / "base" / "pass@k"
    if not base_pass_k.is_dir():
        raise FileNotFoundError(
            f"No base/pass@k directory found under {experiment_dir}. "
            "Pass --model explicitly instead."
        )
    # Search any summary JSON for checkpoint/model_path field
    for json_path in sorted(base_pass_k.rglob("pass_at_k*.json")):
        try:
            with json_path.open() as f:
                data = json.load(f)
            model = data.get("checkpoint") or data.get("model_path")
            if model:
                return model
        except (OSError, json.JSONDecodeError):
            continue
    raise ValueError(
        f"Could not infer base model from summaries in {base_pass_k}. "
        "Pass --model explicitly instead."
    )


def _parse_ks(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _print_summary(output_dir: Path, eval_data: Path, ks: list[int], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"Forgetting probe summary — {label}")
    print(f"Eval data  : {eval_data}")
    print(f"{'='*60}")
    pass_k_dir = output_dir / "pass@k"
    for k in ks:
        summary_path = pass_k_dir / f"pass_at_k_{k}.json"
        if not summary_path.exists():
            summary_path = pass_k_dir / eval_data.stem / "pass_at_k.json"
        if summary_path.exists():
            with summary_path.open() as f:
                data = json.load(f)
            metrics = data.get("pass_at_k", {})
            pct = metrics.get("pass_at_k", float("nan"))
            correct = metrics.get("samples_correct", "?")
            total = metrics.get("total_samples", "?")
            print(f"  pass@{k:<3} = {pct:.1%}  ({correct}/{total} correct)")
        else:
            print(f"  pass@{k}: result not found at {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe for catastrophic forgetting")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Experiment directory (e.g. outputs/rl/.../binary). Required for last-checkpoint mode; optional otherwise.",
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="Run on the original base model. Infers model from --experiment-dir/base/pass@k/ summaries, or use --model.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Explicit model path or HF id.",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=_DEFAULT_EVAL_DATA,
        help=f"Parquet of questions the model originally knew (default: {_DEFAULT_EVAL_DATA})",
    )
    parser.add_argument(
        "--ks",
        default=",".join(str(k) for k in _DEFAULT_KS),
        help="Comma-separated k values (default: 1,16,32)",
    )
    parser.add_argument("--dataset", default="simpleqa")
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gpu-mem-util", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--resp-len", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve() if args.experiment_dir else None
    if experiment_dir is not None and not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    eval_data = args.eval_data.resolve()
    if not eval_data.exists():
        raise FileNotFoundError(f"Eval data not found: {eval_data}")

    ks = _parse_ks(args.ks)
    if not ks:
        raise ValueError("--ks must include at least one value")

    # Resolve model, output directory, and label
    if args.model:
        checkpoint_arg = args.model
        label = Path(args.model).name or args.model
        if experiment_dir:
            output_dir = experiment_dir / "base" / "probe_forgetting"
        else:
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
            output_dir = _REPO_ROOT / "outputs" / "probe_forgetting" / safe
    elif args.base:
        if experiment_dir is not None:
            checkpoint_arg = _infer_base_model(experiment_dir)
        else:
            checkpoint_arg = _DEFAULT_BASE_MODEL
        label = f"base ({checkpoint_arg})"
        if experiment_dir is not None:
            output_dir = experiment_dir / "base" / "probe_forgetting"
        else:
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(checkpoint_arg).name)
            output_dir = _REPO_ROOT / "outputs" / "probe_forgetting" / safe
        print(f"Base model: {checkpoint_arg}")
    else:
        if experiment_dir is None:
            raise ValueError("--experiment-dir is required unless --model or --base is specified")
        last_ckpt = _find_last_checkpoint(experiment_dir)
        print(f"Last checkpoint: {last_ckpt.name}")
        merged = last_ckpt / "merged_hf_model"
        checkpoint_arg = str(merged) if merged.is_dir() else str(last_ckpt)
        output_dir = last_ckpt / "probe_forgetting"
        label = last_ckpt.name

    output_dir.mkdir(parents=True, exist_ok=True)
    pass_k_script = _REPO_ROOT / "scripts" / "pass_at_k.py"

    for k in ks:
        print(f"\n{'='*60}")
        print(f"Running pass@{k} on {eval_data.name}")
        print(f"{'='*60}")
        cmd = [
            sys.executable,
            str(pass_k_script),
            "--checkpoint", checkpoint_arg,
            "--dataset", args.dataset,
            "--eval-data", str(eval_data),
            "--output-dir", str(output_dir),
            "--top-k", str(k),
            "--n-gpus", str(args.n_gpus),
            "--tp-size", str(args.tp_size),
            "--gpu-mem-util", str(args.gpu_mem_util),
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
            "--prompt-len", str(args.prompt_len),
            "--resp-len", str(args.resp_len),
            "--workers", str(args.workers),
        ]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        subprocess.run(cmd, check=True)

    _print_summary(output_dir, eval_data, ks, label)


if __name__ == "__main__":
    main()
