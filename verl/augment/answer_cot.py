"""Generate chain-of-thought answers by grounding SimpleQA rows in RichQA context."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .openai_client import OpenAIClient


SYSTEM_PROMPT = (
    "You write concise chain-of-thought answers. You are given a SimpleQA question and "
    "a richer Q/A pair. Use the rich Q/A only as background knowledge to inform the reasoning — "
    "the final CoT must read as if recalled from memory, with NO references to any provided "
    "context, passage, source, or document. Do NOT write phrases like 'according to the context', "
    "'the passage states', 'the provided information says', 'based on the context', 'as mentioned', etc. "
    "You MAY use first-person recall phrasing like 'I recall that...', 'I remember that...', "
    "or simply state the facts directly.\n\n"
    "Produce exactly one line in the format:\n"
    "\"The question asks [restate question]. [1-2 sentences reasoning from recalled facts toward "
    "the answer]. Therefore, the answer is: [short answer].\"\n"
    "Do not add extra sentences, markdown, or quotes around the output."
)


def _user_prompt(question: str, answer: str, rich_question: str, rich_answer: str) -> str:
    return (
        "[SimpleQA question]\n"
        f"{question}\n\n"
        "[Short answer]\n"
        f"{answer}\n\n"
        "[Background knowledge — use silently, do NOT reference in the output]\n"
        f"Q: {rich_question}\n"
        f"A: {rich_answer}\n\n"
        "Write the single-line chain-of-thought answer now, recalling facts as if from memory."
    )


def _generate_cot(
    client: OpenAIClient,
    *,
    question: str,
    answer: str,
    rich_question: str,
    rich_answer: str,
) -> str:
    return client.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=_user_prompt(question, answer, rich_question, rich_answer),
        temperature=0.2,
    ).strip()


def run_answer_cot(
    *,
    rl_input: str,
    sft_input: str,
    output: str,
    model: str = "gpt-4o-mini",
    batch_size: int = 16,
    max_samples: int | None = None,
    timeout: float = 60.0,
) -> None:
    rl_df = pd.read_parquet(rl_input)
    sft_df = pd.read_parquet(sft_input)

    sft_lookup = {
        row["id"]: (row["question"], row["answer"]) for _, row in sft_df.iterrows()
    }

    if max_samples is not None:
        rl_df = rl_df.head(max_samples).reset_index(drop=True)

    client = OpenAIClient(model, timeout=timeout)

    rows: list[dict] = rl_df.to_dict(orient="records")
    results: list[dict | None] = [None] * len(rows)
    missing: list[int] = []

    def process(idx: int) -> tuple[int, dict | None]:
        row = rows[idx]
        row_id = row["id"]
        if row_id not in sft_lookup:
            return idx, None
        rich_question, rich_answer = sft_lookup[row_id]
        cot = _generate_cot(
            client,
            question=row.get("question", ""),
            answer=row.get("answer", ""),
            rich_question=rich_question,
            rich_answer=rich_answer,
        )
        updated = deepcopy(row)
        updated["answer"] = cot
        return idx, updated

    with ThreadPoolExecutor(max_workers=max(1, batch_size)) as executor:
        futures = {executor.submit(process, i): i for i in range(len(rows))}
        for future in tqdm(as_completed(futures), total=len(futures), desc="answer_cot"):
            idx, updated = future.result()
            if updated is None:
                missing.append(idx)
            else:
                results[idx] = updated

    if missing:
        raise RuntimeError(
            f"{len(missing)} rl rows had no matching sft id (first few indices: {missing[:5]})"
        )

    out_df = pd.DataFrame([r for r in results if r is not None])
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "answer_cot",
        help="Generate chain-of-thought answers grounded in RichQA context",
    )
    p.add_argument("--rl_input", required=True, help="Base SimpleQA rl/train.parquet")
    p.add_argument("--sft_input", required=True, help="RichQA sft/train.parquet")
    p.add_argument("--output", required=True, help="Output parquet path")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--timeout", type=float, default=60.0)
    p.set_defaults(
        _handler=lambda args: run_answer_cot(
            rl_input=args.rl_input,
            sft_input=args.sft_input,
            output=args.output,
            model=args.model,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            timeout=args.timeout,
        )
    )
