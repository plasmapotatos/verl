#!/usr/bin/env python3
"""Mirror data/simpleqa/augment/refusal -> data/simpleqa/augment/refusal_clean.

Identical rows (ids, answers, ground_truth, extra_info, target, ...). The only
mutation is appending an IDK suffix to `question` and to the user-content inside
`prompt`. This makes refusal_clean the rich-QA analog of the existing
partition/refusal_clean dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal")
DST = Path("/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal_clean")
SUFFIX = ' If you are unsure, just say "I don\'t know."'


def append_suffix_prompt(prompt):
    if isinstance(prompt, np.ndarray):
        out = []
        for msg in prompt:
            msg = dict(msg)
            if msg.get("role") == "user":
                msg["content"] = msg["content"] + SUFFIX
            out.append(msg)
        return np.array(out, dtype=object)
    if isinstance(prompt, list):
        out = []
        for msg in prompt:
            msg = dict(msg)
            if msg.get("role") == "user":
                msg["content"] = msg["content"] + SUFFIX
            out.append(msg)
        return out
    raise TypeError(f"unexpected prompt type: {type(prompt)}")


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "question" in df.columns:
        df["question"] = df["question"].astype(str) + SUFFIX
    if "prompt" in df.columns:
        df["prompt"] = df["prompt"].apply(append_suffix_prompt)
    return df


def main() -> None:
    parquets = []
    for root, _dirs, files in os.walk(SRC):
        for f in sorted(files):
            if f.endswith(".parquet"):
                parquets.append(Path(root) / f)
    print(f"Found {len(parquets)} parquet files under {SRC}")

    for src_path in parquets:
        rel = src_path.relative_to(SRC)
        dst_path = DST / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(src_path)
        df_out = transform(df)
        df_out.to_parquet(dst_path, index=False)
        print(f"  {len(df_out):>5} rows -> {dst_path}")

    # Also copy non-parquet metadata files (metadata.txt, metrics json, etc.)
    for root, _dirs, files in os.walk(SRC):
        for f in sorted(files):
            if f.endswith(".parquet"):
                continue
            src = Path(root) / f
            rel = src.relative_to(SRC)
            dst = DST / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            print(f"  copied -> {dst}")

    (DST / "README.md").write_text(
        "# refusal_clean (rich-QA augment)\n\n"
        "Identical to data/simpleqa/augment/refusal/, except every question and\n"
        "user-prompt has the suffix appended:\n\n"
        f'    {SUFFIX!r}\n\n'
        "Same IDs, answers, ground_truth, extra_info, target. Built by\n"
        "scripts/make_refusal_clean_augment.py.\n"
    )
    print(f"\nWrote tree under {DST}")


if __name__ == "__main__":
    main()
