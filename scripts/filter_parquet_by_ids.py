#!/usr/bin/env python
"""Filter parquet A to keep only rows whose sample_id appears in parquet B.

Usage:
    python scripts/filter_parquet_by_ids.py <a.parquet> <b.parquet> [-o out.parquet]

sample_id is read from extra_info.sample_id, falling back to a top-level
sample_id column.
"""

import argparse
from pathlib import Path

import pandas as pd


def get_ids(df: pd.DataFrame) -> pd.Series:
    if "extra_info" in df.columns:
        ids = df["extra_info"].apply(
            lambda x: x.get("sample_id") if isinstance(x, dict) else None
        )
        if ids.notna().any():
            return ids
    if "sample_id" in df.columns:
        return df["sample_id"]
    raise KeyError("No sample_id found in extra_info or top-level columns")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path, help="Parquet to filter")
    p.add_argument("b", type=Path, help="Parquet whose ids define the keep-set")
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args()

    a = pd.read_parquet(args.a)
    b = pd.read_parquet(args.b)

    keep = set(get_ids(b).dropna())
    a_ids = get_ids(a)
    mask = a_ids.isin(keep)

    out = args.output or args.a.parent / f"{args.a.stem}_filtered.parquet"
    a[mask].to_parquet(out, index=False)
    print(f"{args.a} ({len(a)}) -> {out} ({int(mask.sum())}; dropped {len(a) - int(mask.sum())})")


if __name__ == "__main__":
    main()
