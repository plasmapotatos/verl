#!/usr/bin/env python3
"""
Prepare factual_anchor data splits.

Two modes:
  filter       -- filter both datasets to their ID intersection and overwrite
  align-origqa -- align origqa to factual_anchor splits (run after make_eval_subset.py)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def save(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Wrote {len(df)} rows → {path}")


def filter_mode(args: argparse.Namespace) -> None:
    fa = pd.read_parquet(args.factual_anchor)
    oq = pd.read_parquet(args.origqa)

    fa_ids = set(fa["id"].astype(str))
    oq_ids = set(oq["id"].astype(str))
    common_ids = fa_ids & oq_ids

    print(f"factual_anchor: {len(fa)} rows")
    print(f"origqa:         {len(oq)} rows")
    print(f"intersection:   {len(common_ids)} IDs")
    print(f"dropped from factual_anchor: {len(fa) - len(common_ids)}")
    print(f"dropped from origqa:         {len(oq) - len(common_ids)}")

    fa_filtered = fa[fa["id"].astype(str).isin(common_ids)].reset_index(drop=True)
    oq_filtered = oq[oq["id"].astype(str).isin(common_ids)].reset_index(drop=True)

    save(fa_filtered, args.factual_anchor)
    save(oq_filtered, args.origqa)


def align_origqa_mode(args: argparse.Namespace) -> None:
    oq = pd.read_parquet(args.origqa)
    oq_indexed = oq.set_index(oq["id"].astype(str))

    out_dir = args.out_dir
    oq_stem = Path(args.origqa).stem  # e.g. simpleqa_rich_sft_train_origqa_factual_anchor

    splits = [
        (args.fa_eval,  "_frac0.1_eval"),
        (args.fa_train, "_frac0.9_train"),
        (args.fa_passk, "_frac0.9_train_frac0.1"),
    ]

    for fa_split_path, oq_suffix in splits:
        fa_split = pd.read_parquet(fa_split_path)
        split_ids = fa_split["id"].astype(str).tolist()

        missing = [sid for sid in split_ids if sid not in oq_indexed.index]
        if missing:
            print(f"WARNING: {len(missing)} IDs from {fa_split_path} not found in origqa")

        aligned = oq_indexed.loc[[sid for sid in split_ids if sid in oq_indexed.index]].reset_index(drop=True)
        out_path = str(Path(out_dir) / f"{oq_stem}{oq_suffix}.parquet")
        save(aligned, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    # filter mode
    p_filter = sub.add_parser("filter", help="Filter both datasets to their ID intersection")
    p_filter.add_argument("--factual-anchor", required=True, help="Path to factual_anchor parquet (overwritten)")
    p_filter.add_argument("--origqa", required=True, help="Path to origqa parquet (overwritten)")

    # align-origqa mode
    p_align = sub.add_parser("align-origqa", help="Align origqa splits to factual_anchor splits by ID")
    p_align.add_argument("--origqa", required=True, help="Path to filtered origqa base parquet")
    p_align.add_argument("--fa-eval", required=True, help="factual_anchor eval split (frac0.1)")
    p_align.add_argument("--fa-train", required=True, help="factual_anchor train split (frac0.9)")
    p_align.add_argument("--fa-passk", required=True, help="factual_anchor pass@k split (frac0.9_train_frac0.1)")
    p_align.add_argument("--out-dir", required=True, help="Output directory for origqa splits")

    args = parser.parse_args()

    if args.mode == "filter":
        filter_mode(args)
    elif args.mode == "align-origqa":
        align_origqa_mode(args)


if __name__ == "__main__":
    main()
