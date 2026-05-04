from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COT_SUFFIX = (
    "\n\nThink step by step. End your response with a single line of the form:\n"
    "Answer: <your final answer>"
)


def convert(in_path: str, out_path: str) -> None:
    df = pd.read_parquet(in_path)
    for col in ("id", "question", "answer"):
        if col not in df.columns:
            raise ValueError(f"Input parquet missing required column: {col}")

    questions_cot = df["question"].astype(str) + COT_SUFFIX

    prompts = np.empty(len(df), dtype=object)
    for i, q in enumerate(questions_cot):
        prompts[i] = [{"role": "user", "content": q}]

    out = pd.DataFrame(
        {
            "id": df["id"].astype(str).values,
            "question": questions_cot.values,
            "answer": df["answer"].astype(str).values,
            "prompt": prompts,
        }
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} rows -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
