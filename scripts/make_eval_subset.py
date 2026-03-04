#!/usr/bin/env python3
"""Sample a fraction of rows from a parquet file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def _default_output_path(input_path: str, fraction: float) -> str:
    base = Path(input_path)
    suffix = f"_frac{fraction:g}"
    return str(base.with_name(base.stem + suffix + base.suffix))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet path (defaults to input with _fracX suffix)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Fraction of rows to sample (0 < fraction <= 1)",
    )
    args = parser.parse_args()

    if args.fraction <= 0 or args.fraction > 1:
        raise ValueError("--fraction must be in (0, 1]")

    input_path = args.input
    output_path = args.output or _default_output_path(input_path, args.fraction)

    df = pd.read_parquet(input_path)
    sampled = df.sample(frac=args.fraction, random_state=args.seed)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    sampled.to_parquet(output_path, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(sampled)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
