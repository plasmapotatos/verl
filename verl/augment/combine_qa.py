"""Combine k Q/A pairs sharing an id into one merged Q/A using an LLM."""

from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .openai_client import OpenAIClient


PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "combine_qa.txt"


def _load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _user_prompt(pairs: list[tuple[str, str]]) -> str:
    lines = ["Combine the following Q/A pairs into one merged Q/A:\n"]
    for i, (q, a) in enumerate(pairs, 1):
        lines.append(f"Pair {i}:")
        lines.append(f"  Q: {q}")
        lines.append(f"  A: {a}")
    lines.append("\nRespond with JSON: {\"question\": ..., \"answer\": ...}")
    return "\n".join(lines)


def _parse_response(text: str) -> tuple[str, str]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    obj = json.loads(text)
    return str(obj["question"]).strip(), str(obj["answer"]).strip()


def _sample_distinct_combinations(
    group_size: int, k: int, n: int, rng: random.Random
) -> list[tuple[int, ...]]:
    """Return up to n distinct k-combinations of range(group_size).

    If C(group_size, k) < n, returns all available combinations rather than skipping.
    """
    if group_size < k:
        return []
    all_combos = list(combinations(range(group_size), k))
    rng.shuffle(all_combos)
    return all_combos[:n]


def run_combine_qa(
    *,
    input_path: str,
    output: str,
    k: int = 2,
    n: int = 1,
    model: str = "gpt-4o-mini",
    batch_size: int = 16,
    max_groups: int | None = None,
    seed: int = 0,
    timeout: float = 60.0,
) -> None:
    df = pd.read_parquet(input_path)
    if "id" not in df.columns or "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("Input parquet must have columns: id, question, answer")

    rng = random.Random(seed)
    tasks: list[tuple[str, int, list[tuple[str, str]], list[int]]] = []
    skipped_size = 0
    for gid, sub in df.groupby("id"):
        group_size = len(sub)
        if group_size < k:
            skipped_size += 1
            continue
        combos = _sample_distinct_combinations(group_size, k, n, rng)
        if not combos:
            skipped_size += 1
            continue
        sub_reset = sub.reset_index(drop=True)
        for draw_idx, idxs in enumerate(combos):
            idxs_list = list(idxs)
            rows = sub_reset.iloc[idxs_list]
            pairs = [(str(r["question"]), str(r["answer"])) for _, r in rows.iterrows()]
            tasks.append((gid, draw_idx, pairs, idxs_list))

    if max_groups is not None:
        kept_gids: list[str] = []
        kept_set: set[str] = set()
        for gid, _, _ in tasks:
            if gid not in kept_set:
                kept_set.add(gid)
                kept_gids.append(gid)
                if len(kept_gids) >= max_groups:
                    break
        tasks = [t for t in tasks if t[0] in kept_set]

    print(
        f"{len(tasks)} merge tasks across {len({t[0] for t in tasks})} groups "
        f"(skipped {skipped_size} for size<k or no combos)"
    )

    client = OpenAIClient(model, timeout=timeout)
    results: list[dict | None] = [None] * len(tasks)

    def process(idx: int) -> tuple[int, dict | None]:
        gid, draw_idx, pairs, idxs_list = tasks[idx]
        try:
            text = client.generate(
                system_prompt=_load_system_prompt(),
                user_prompt=_user_prompt(pairs),
                temperature=0.2,
            )
            q, a = _parse_response(text)
        except Exception as exc:
            print(f"[combine_qa] group {gid} draw {draw_idx} failed: {exc}")
            return idx, None
        return idx, {
            "id": gid,
            "draw_idx": draw_idx,
            "question": q,
            "answer": a,
            "source_subq_indices": idxs_list,
            "source_questions": [p[0] for p in pairs],
            "source_answers": [p[1] for p in pairs],
        }

    with ThreadPoolExecutor(max_workers=max(1, batch_size)) as executor:
        futures = {executor.submit(process, i): i for i in range(len(tasks))}
        for future in tqdm(as_completed(futures), total=len(futures), desc="combine_qa"):
            idx, row = future.result()
            results[idx] = row

    out_rows = [r for r in results if r is not None]
    out_df = pd.DataFrame(
        out_rows,
        columns=[
            "id",
            "draw_idx",
            "question",
            "answer",
            "source_subq_indices",
            "source_questions",
            "source_answers",
        ],
    )
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "combine_qa",
        help="Combine k Q/A pairs sharing an id into one merged Q/A via LLM",
    )
    p.add_argument("--input", required=True, help="Input parquet with id/question/answer")
    p.add_argument("--output", required=True, help="Output parquet path")
    p.add_argument("--k", type=int, default=2, help="Samples per id group (default 2)")
    p.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of distinct k-combinations to draw per id; "
             "skip the id if C(group_size, k) < n",
    )
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_groups", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=60.0)
    p.set_defaults(
        _handler=lambda args: run_combine_qa(
            input_path=args.input,
            output=args.output,
            k=args.k,
            n=args.n,
            model=args.model,
            batch_size=args.batch_size,
            max_groups=args.max_groups,
            seed=args.seed,
            timeout=args.timeout,
        )
    )
