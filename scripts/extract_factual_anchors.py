#!/usr/bin/env python3
"""Extract bracketed factual anchors from a parquet answer column.

Example:
    apptainer exec /work/hdd/bbsg/twei2/rl/torch2501.sif python scripts/extract_factual_anchors.py \
        --input data/simpleqa/partition/factual_anchor/sft/train.parquet \
        --output data/simpleqa/factual_anchor_bank.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def extract_anchors(text: str) -> list[str]:
    """Return bracketed anchor strings without brackets."""
    if not text:
        return []
    return [m.strip() for m in re.findall(r"\[([^\[\]]+)\]", str(text)) if m.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--output", required=True, help="Output .txt path (one anchor per line)")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if "answer" not in df.columns:
        raise KeyError(f'Expected column "answer" in {args.input}')

    anchors: set[str] = set()
    for answer in df["answer"].tolist():
        for anchor in extract_anchors(answer):
            anchors.add(anchor)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_anchors = sorted(anchors)
    output_path.write_text("\n".join(sorted_anchors) + "\n", encoding="utf-8")

    print(f"Input rows: {len(df)}")
    print(f"Unique anchors: {len(sorted_anchors)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
