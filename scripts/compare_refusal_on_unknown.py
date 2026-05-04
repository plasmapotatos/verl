#!/usr/bin/env python3
"""Compare refusal behavior of base / SFT / RL checkpoints on the unknown_test split.

Loads each model in turn, runs greedy generation on every prompt in the parquet,
flags responses as refusals via the project's _NOT_ATTEMPTED_PATTERNS heuristic,
and writes per-model JSONL plus a summary table.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from verl.utils.reward_score.simpleqa import _is_not_attempted  # noqa: E402


MODELS = [
    ("base_qwen", "Qwen/Qwen2.5-3B-Instruct"),
    (
        "sft_richqa_refusal_50_50",
        "/work/hdd/bbsg/twei2/rl/verl/outputs/sft/simpleqa_rich_sft_refusal_partition_50_50/sft_lr1.5e-4_epmax30_seed1/global_step_1290/merged_hf_model",
    ),
    (
        "rl_richqa_grpo_refusal_50_50",
        "/work/hdd/bbsg/twei2/rl/verl/outputs/rl/richqa_grpo_refusal_50_50/binary/global_step_680/merged_hf_model",
    ),
]


def build_chat_prompts(tokenizer, questions: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]


def run_model(label: str, model_path: str, questions: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
    print(f"\n=== {label} :: {model_path} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    prompts = build_chat_prompts(tokenizer, questions)
    outputs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_only = gen[:, enc["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        outputs.extend(decoded)
        print(f"  [{label}] {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/unknown_test.parquet",
    )
    parser.add_argument(
        "--out-dir",
        default="/work/hdd/bbsg/twei2/rl/verl/outputs/analysis/refusal_unknown_test",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on #prompts (0=all)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit)
    questions = df["question"].astype(str).tolist()
    answers = df["answer"].astype(str).tolist()
    ids = df["id"].astype(str).tolist()
    print(f"Loaded {len(questions)} prompts from {args.data}")

    summary = []
    for label, path in MODELS:
        responses = run_model(label, path, questions, args.batch_size, args.max_new_tokens)
        refusals = [_is_not_attempted(r) for r in responses]
        rate = sum(refusals) / len(refusals) if refusals else 0.0

        out_file = out_dir / f"{label}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for qid, q, a, r, ref in zip(ids, questions, answers, responses, refusals):
                f.write(json.dumps({
                    "id": qid,
                    "question": q,
                    "gold_answer": a,
                    "response": r,
                    "is_refusal": bool(ref),
                }, ensure_ascii=False) + "\n")
        print(f"  -> wrote {out_file}  refusal_rate={rate:.3f}  ({sum(refusals)}/{len(refusals)})")
        summary.append({"model": label, "path": path, "n": len(refusals), "refusals": int(sum(refusals)), "refusal_rate": rate})

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n=== Refusal rates on unknown_test ===")
    for row in summary:
        print(f"  {row['model']:<32s}  {row['refusals']:>4d}/{row['n']:<4d}  ({row['refusal_rate']:.3f})")
    print(f"\nWrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
