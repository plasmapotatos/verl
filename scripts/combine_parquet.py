# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility to concatenate and optionally shuffle a series of parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine multiple parquet shards into one dataset")
    parser.add_argument(
        "--input_paths",
        nargs="+",
        required=True,
        help="List of parquet files to concatenate in order.",
    )
    parser.add_argument("--output_dir", help="Target directory to write the joined parquet file.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffle is enabled.",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Keep the original order after concatenation.",
    )
    parser.set_defaults(shuffle=True)
    return parser.parse_args()


def _resolve_output_dir(input_paths: Sequence[Path], explicit_output: str | None) -> Path:
    if explicit_output:
        return Path(explicit_output).expanduser()
    first_path = input_paths[0]
    return first_path.parent / f"{first_path.stem}_combined"


def main() -> None:
    args = _parse_args()
    if not args.input_paths:
        raise SystemExit("--input_paths is required and must contain at least one parquet file")

    input_paths = [Path(path).expanduser() for path in args.input_paths]
    for path in input_paths:
        if not path.exists():
            raise SystemExit(f"input path does not exist: {path}")

    target_dir = _resolve_output_dir(input_paths, args.output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "combined.parquet"
    print(f"Combining {len(input_paths)} files into {target_file} (shuffle={args.shuffle})")

    frames: list[pd.DataFrame] = []
    for path in input_paths:
        frames.append(pd.read_parquet(path))

    if not frames:
        raise SystemExit("No data loaded from input paths")

    combined = pd.concat(frames, ignore_index=True)
    if args.shuffle:
        combined = combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    combined.to_parquet(target_file)
    print(f"wrote {len(combined)} rows to {target_file}")


if __name__ == "__main__":
    main()
