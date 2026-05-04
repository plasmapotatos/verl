#!/usr/bin/env python
"""Remove factual-anchor brackets ([...] -> ...) from parquet files.

Usage:
    python scripts/debracket_parquet.py <input.parquet> [<input2.parquet> ...]

For each input file, writes to {stem}_unbracketed.parquet in the same directory.
Debracketing is applied to: question, answer, prompt (chat messages), and
reward_model.ground_truth.
"""

import copy
import re
import sys
from pathlib import Path

import pandas as pd


def debracket(text: str) -> str:
    """Remove square brackets around factual anchors: [foo] -> foo."""
    if not isinstance(text, str):
        return text
    return re.sub(r"\[([^\[\]]+)\]", r"\1", text)


def debracket_prompt(prompt):
    """Debracket the 'content' field in each message of a prompt list/array."""
    try:
        iter(prompt)
    except TypeError:
        return prompt
    out = []
    for msg in prompt:
        msg = dict(msg) if hasattr(msg, "items") else msg
        if isinstance(msg, dict) and "content" in msg:
            msg["content"] = debracket(msg["content"])
        out.append(msg)
    return out


def debracket_reward_model(rm):
    """Debracket the ground_truth inside the reward_model dict."""
    if not isinstance(rm, dict):
        return rm
    rm = dict(rm)
    if "ground_truth" in rm:
        rm["ground_truth"] = debracket(rm["ground_truth"])
    return rm


def process_file(path: Path):
    df = pd.read_parquet(path)

    if "question" in df.columns:
        df["question"] = df["question"].apply(debracket)
    if "answer" in df.columns:
        df["answer"] = df["answer"].apply(debracket)
    if "prompt" in df.columns:
        df["prompt"] = df["prompt"].apply(debracket_prompt)
    if "reward_model" in df.columns:
        df["reward_model"] = df["reward_model"].apply(debracket_reward_model)

    out_path = path.parent / f"{path.stem}_unbracketed.parquet"
    df.to_parquet(out_path, index=False)
    print(f"{path} -> {out_path}  ({len(df)} rows)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        process_file(Path(arg))


if __name__ == "__main__":
    main()
