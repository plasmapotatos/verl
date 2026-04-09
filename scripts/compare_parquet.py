#!/usr/bin/env python3
"""Compare two parquet files by ID and per-column values."""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two parquet files by ID and column values")
    parser.add_argument("a", help="First parquet file")
    parser.add_argument("b", help="Second parquet file")
    parser.add_argument("--id-col", default="id", help="ID column name (default: id)")
    parser.add_argument("--columns", nargs="*", help="Columns to compare (default: all shared columns)")
    parser.add_argument("--show-diffs", type=int, default=5, help="Number of example diffs to show per column (default: 5)")
    args = parser.parse_args()

    df_a = pd.read_parquet(args.a)
    df_b = pd.read_parquet(args.b)

    print(f"A: {args.a}  ({len(df_a)} rows, {len(df_a.columns)} cols)")
    print(f"B: {args.b}  ({len(df_b)} rows, {len(df_b.columns)} cols)")
    print()

    # Column comparison
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    only_a = cols_a - cols_b
    only_b = cols_b - cols_a
    shared = sorted(cols_a & cols_b)

    if only_a:
        print(f"Columns only in A: {sorted(only_a)}")
    if only_b:
        print(f"Columns only in B: {sorted(only_b)}")
    print(f"Shared columns: {shared}")
    print()

    # ID comparison
    id_col = args.id_col
    if id_col not in df_a.columns or id_col not in df_b.columns:
        print(f"ERROR: '{id_col}' column missing from one or both files", file=sys.stderr)
        sys.exit(1)

    ids_a = set(df_a[id_col].astype(str))
    ids_b = set(df_b[id_col].astype(str))
    only_in_a = ids_a - ids_b
    only_in_b = ids_b - ids_a
    common = ids_a & ids_b

    print(f"IDs in A: {len(ids_a)}")
    print(f"IDs in B: {len(ids_b)}")
    print(f"Common IDs: {len(common)}")
    if only_in_a:
        examples = sorted(only_in_a)[:5]
        print(f"Only in A: {len(only_in_a)} (e.g. {examples})")
    if only_in_b:
        examples = sorted(only_in_b)[:5]
        print(f"Only in B: {len(only_in_b)} (e.g. {examples})")
    print()

    if not common:
        print("No common IDs to compare.")
        sys.exit(0)

    # Align by ID for per-column comparison
    df_a = df_a.drop_duplicates(subset=id_col, keep="last").set_index(df_a[id_col].astype(str))
    df_b = df_b.drop_duplicates(subset=id_col, keep="last").set_index(df_b[id_col].astype(str))
    common_ids = sorted(common)
    df_a = df_a.loc[common_ids]
    df_b = df_b.loc[common_ids]

    compare_cols = args.columns if args.columns else [c for c in shared if c != id_col]
    all_match = True

    for col in compare_cols:
        if col not in df_a.columns or col not in df_b.columns:
            print(f"  {col}: SKIPPED (missing from one file)")
            continue

        a_vals = df_a[col].reset_index(drop=True)
        b_vals = df_b[col].reset_index(drop=True)

        try:
            eq = a_vals.astype(str) == b_vals.astype(str)
        except Exception:
            eq = pd.Series([False] * len(a_vals))

        n_match = eq.sum()
        n_diff = len(eq) - n_match

        if n_diff == 0:
            print(f"  {col}: MATCH ({n_match}/{len(eq)})")
        else:
            all_match = False
            print(f"  {col}: DIFF ({n_diff}/{len(eq)} differ)")
            if args.show_diffs > 0:
                diff_idx = eq[~eq].index[:args.show_diffs]
                for idx in diff_idx:
                    row_id = common_ids[idx]
                    val_a = str(a_vals.iloc[idx])[:80]
                    val_b = str(b_vals.iloc[idx])[:80]
                    print(f"    id={row_id}: A={val_a!r}")
                    print(f"    {'':>{len('id=' + row_id)}}  B={val_b!r}")

    print()
    if all_match and not only_in_a and not only_in_b:
        print("RESULT: Files are equivalent.")
    elif all_match:
        print("RESULT: Common rows match, but ID sets differ.")
    else:
        print("RESULT: Files differ.")


if __name__ == "__main__":
    main()
