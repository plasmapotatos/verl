"""Cap the number of samples per ID in a parquet dataset.

Randomly samples up to k rows per unique ID, preserving the original
column order and dtypes.

Usage:
    python scripts/cap_samples_per_id.py INPUT.parquet OUTPUT.parquet --k 7 [--seed 42]
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Cap samples per ID in a parquet file")
    parser.add_argument("input", help="Input parquet file")
    parser.add_argument("output", help="Output parquet file")
    parser.add_argument("--k", type=int, required=True, help="Max samples per ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    original_len = len(df)
    unique_ids = df["id"].nunique()

    idx = (
        df.groupby("id", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), args.k), random_state=args.seed))
        .index
    )
    capped = df.loc[idx].reset_index(drop=True)

    ids_affected = (df.groupby("id").size() > args.k).sum()

    print(f"Input:  {original_len} rows, {unique_ids} unique IDs")
    print(f"Cap:    k={args.k}")
    print(f"IDs affected (had > k): {ids_affected}")
    print(f"Output: {len(capped)} rows ({len(capped)/original_len*100:.1f}% retained)")
    print(f"Saved to: {args.output}")

    capped.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
