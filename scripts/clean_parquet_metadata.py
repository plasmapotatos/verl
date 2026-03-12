#!/usr/bin/env python3
"""Remove top-level extra_info column from a parquet dataset in place."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop the extra_info and metadata column from a parquet file")
    parser.add_argument("parquet", type=Path, help="Path to the parquet file to clean")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Keep a .bak copy before overwriting",
    )
    args = parser.parse_args()

    if not args.parquet.exists():
        raise FileNotFoundError(args.parquet)

    if args.backup:
        backup_path = args.parquet.with_suffix(args.parquet.suffix + ".bak")
        shutil.copy2(args.parquet, backup_path)

    df = pd.read_parquet(args.parquet)

    df = df.drop(columns="extra_info") if "extra_info" in df.columns else df
    df = df.drop(columns="metadata") if "metadata" in df.columns else df
    df.to_parquet(args.parquet)
    print(f"Updated {args.parquet} (removed extra_info and metadata columns)")


if __name__ == "__main__":
    main()
