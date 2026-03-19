#!/usr/bin/env python3
"""Generate k samples and compute pass@k for a dataset."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from verl.eval.runner import run as run_eval


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sanitize_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _default_run_name(checkpoint: str, k: int, temperature: float, top_p: float) -> str:
    ckpt_name = _sanitize_label(Path(checkpoint).name or "model")
    return f"{ckpt_name}_k{k}_t{temperature}_p{top_p}"


def _sync_hf_aux_files(merged_dir: Path, aux_dir: Path) -> None:
    merged_dir.mkdir(parents=True, exist_ok=True)
    preproc_src = aux_dir / "preprocessor_config.json"
    chat_template_src = aux_dir / "chat_template.json"

    if preproc_src.exists():
        (merged_dir / preproc_src.name).write_bytes(preproc_src.read_bytes())
    if chat_template_src.exists():
        (merged_dir / chat_template_src.name).write_bytes(chat_template_src.read_bytes())


def _resolve_model_path(
    checkpoint: str,
    output_dir: Path,
    merge_if_needed: bool,
    hf_aux_src_dir: Path,
) -> str:
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        return checkpoint

    merged_candidate = ckpt_path / "merged_hf_model"
    if merged_candidate.is_dir():
        _sync_hf_aux_files(merged_candidate, hf_aux_src_dir)
        return str(merged_candidate)

    if (ckpt_path / "config.json").exists():
        _sync_hf_aux_files(ckpt_path, hf_aux_src_dir)
        return str(ckpt_path)

    if not merge_if_needed:
        raise ValueError(f"Checkpoint does not look like a HF model dir: {checkpoint}")

    merged_dir = output_dir / "merged_hf_model"
    if not merged_dir.exists():
        cmd = [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(ckpt_path),
            "--target_dir",
            str(merged_dir),
        ]
        subprocess.run(cmd, check=True)

    _sync_hf_aux_files(merged_dir, hf_aux_src_dir)
    return str(merged_dir)


def _compute_pass_at_k(rows: List[Dict[str, object]]) -> Dict[str, object]:
    total = len(rows)
    correct = 0
    all_not_attempted = 0
    failed = 0

    for row in rows:
        graders = row.get("graders", []) if isinstance(row, dict) else []
        evaluations = []
        for grader in graders or []:
            if isinstance(grader, dict):
                evaluations.append(grader.get("evaluation", ""))

        if not evaluations:
            failed += 1
            continue

        if any(eval_label == "correct" for eval_label in evaluations):
            correct += 1
        else:
            if all(eval_label == "not_attempted" for eval_label in evaluations):
                all_not_attempted += 1

    pass_at_k = correct / total if total else 0.0
    return {
        "total_samples": total,
        "samples_correct": correct,
        "samples_incorrect": total - correct,
        "samples_all_not_attempted": all_not_attempted,
        "samples_failed": failed,
        "pass_at_k": pass_at_k,
    }


def _load_existing_summary(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    return payload if isinstance(payload, dict) else None


def _matches_existing_run(summary: Dict[str, object], checkpoint: str, eval_path: Path, k: int) -> bool:
    existing_eval = summary.get("eval_data")
    existing_k = summary.get("k")
    existing_checkpoint = summary.get("checkpoint")
    existing_model_path = summary.get("model_path")

    eval_matches = existing_eval == str(eval_path)
    k_matches = existing_k == k
    checkpoint_matches = existing_checkpoint == checkpoint or existing_model_path == checkpoint
    return eval_matches and k_matches and checkpoint_matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path or HF model id")
    parser.add_argument("--dataset", default="simpleqa", help="Dataset name for eval")
    parser.add_argument("--eval-data", required=True, help="Eval parquet path")
    parser.add_argument("--output-dir", required=True, help="Output directory for pass@k artifacts")
    parser.add_argument("--top-k", type=int, required=True, help="Number of samples per prompt (k)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature")
    parser.add_argument("--prompt-len", type=int, default=2048, help="Prompt length")
    parser.add_argument("--resp-len", type=int, default=1024, help="Response length")
    parser.add_argument("--n-gpus", type=int, default=1, help="GPUs for generation")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-mem-util", type=float, default=0.8, help="GPU memory utilization")
    parser.add_argument("--cache-dir", default=None, help="Dataset cache directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows")
    parser.add_argument("--use-judge", default=True, action="store_true", help="Use OpenAI judge")
    parser.add_argument("--workers", type=int, default=32, help="Parallel workers for eval")
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable merging FSDP checkpoints to HF format",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be a positive integer")

    eval_path = Path(args.eval_data)
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval parquet not found: {eval_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pass_k_dir = output_dir / "pass@k"
    pass_k_dir.mkdir(parents=True, exist_ok=True)

    hf_aux_dir = _repo_root() / "configs" / "hf_aux_files"
    model_path = _resolve_model_path(args.checkpoint, output_dir, not args.no_merge, hf_aux_dir)

    gen_out = pass_k_dir / f"generations_{args.top_k}.parquet"
    if not gen_out.exists():
        cmd = [
            sys.executable,
            "-m",
            "verl.trainer.main_generation",
            "trainer.nnodes=1",
            f"trainer.n_gpus_per_node={args.n_gpus}",
            f"data.path={eval_path}",
            "data.prompt_key=prompt",
            f"data.n_samples={args.top_k}",
            f"data.output_path={gen_out}",
            f"model.path={model_path}",
            "+model.trust_remote_code=True",
            f"rollout.temperature={args.temperature}",
            f"rollout.top_p={args.top_p}",
            f"rollout.prompt_length={args.prompt_len}",
            f"rollout.response_length={args.resp_len}",
            f"rollout.tensor_model_parallel_size={args.tp_size}",
            f"rollout.gpu_memory_utilization={args.gpu_mem_util}",
        ]
        subprocess.run(cmd, check=True)

    eval_out = pass_k_dir / f"eval_{args.top_k}.json"
    summary_path = pass_k_dir / f"pass_at_k_{args.top_k}.json"
    existing_summary = _load_existing_summary(summary_path)
    if (
        existing_summary is not None
        and eval_out.exists()
        and _matches_existing_run(existing_summary, args.checkpoint, eval_path, args.top_k)
    ):
        print(f"Skipping pass@k (already exists): {summary_path}")
        return

    eval_result = run_eval(
        dataset_name=args.dataset,
        input_parquet=str(gen_out),
        output_json=str(eval_out),
        use_judge=args.use_judge,
        cache_dir=args.cache_dir,
        limit=args.limit,
        workers=args.workers,
    )

    rows = eval_result.get("rows", []) if isinstance(eval_result, dict) else []
    pass_metrics = _compute_pass_at_k(rows)

    summary = {
        "checkpoint": args.checkpoint,
        "model_path": model_path,
        "dataset": args.dataset,
        "eval_data": str(eval_path),
        "k": args.top_k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "generation_parquet": str(gen_out),
        "eval_json": str(eval_out),
        "response_metrics": eval_result.get("metrics", {}),
        "pass_at_k": pass_metrics,
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
