#!/usr/bin/env python3
"""Copy answers from a reference parquet into the `target` field of another parquet."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def _default_output_path(base_path: Path) -> Path:
    return base_path.with_name(base_path.stem + "_with_target" + base_path.suffix)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate `target` using answers from a reference parquet.")
    parser.add_argument(
        "--primary",
        required=True,
        help="Parquet that should receive the target answers (will not be modified in place).",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Parquet that supplies answers keyed by the `id` column.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the updated parquet (defaults to <primary>_with_target.parquet).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the primary parquet instead of writing a new file.",
    )
    args = parser.parse_args()

    primary_path = Path(args.primary).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve()

    if not primary_path.exists() or not primary_path.is_file():
        raise FileNotFoundError(f"Primary parquet not found: {primary_path}")
    if not reference_path.exists() or not reference_path.is_file():
        raise FileNotFoundError(f"Reference parquet not found: {reference_path}")

    primary_df = pd.read_parquet(primary_path)
    reference_df = pd.read_parquet(reference_path)

    if "id" not in primary_df.columns:
        raise ValueError("Primary parquet is missing the required 'id' column")
    if "id" not in reference_df.columns:
        raise ValueError("Reference parquet is missing the required 'id' column")
    if "answer" not in reference_df.columns:
        raise ValueError("Reference parquet is missing the required 'answer' column")

    answer_map = reference_df.drop_duplicates(subset="id", keep="last").set_index("id")["answer"]
    primary_df["target"] = primary_df["id"].map(answer_map)

    missing = primary_df["target"].isna().sum()
    print(f"Failed to fill target for {missing} rows")

    if args.in_place:
        if args.output:
            raise ValueError("--output cannot be combined with --in-place")
        primary_df.to_parquet(primary_path, index=False)
        print(f"Updated in place: {primary_path}")
    else:
        output_path = Path(args.output) if args.output else _default_output_path(primary_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        primary_df.to_parquet(output_path, index=False)
        print(f"Wrote: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(type(exc).__name__, exc, file=sys.stderr)
        sys.exit(1)
