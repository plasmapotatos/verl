from __future__ import annotations

import argparse
import sys

from .datasets import list_datasets
from .runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VERL evaluation runner")
    parser.add_argument("--dataset", required=False, help="Dataset name")
    parser.add_argument("--input", required=False, help="Input predictions parquet")
    parser.add_argument("--output", required=False, help="Output JSON path")
    parser.add_argument("--use-judge", action="store_true", help="Use OpenAI judge")
    parser.add_argument("--cache-dir", help="Dataset cache directory")
    parser.add_argument("--limit", type=int, help="Limit number of rows")
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of parallel workers for evaluation (default: 32)",
    )
    parser.add_argument("--list-datasets", action="store_true", help="List datasets")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_datasets:
        for name in list_datasets():
            print(name)
        return 0

    missing = [flag for flag in ("--dataset", "--input", "--output") if getattr(args, flag[2:]) is None]
    if missing:
        parser.error("Missing required arguments: " + ", ".join(missing))

    try:
        run(
            dataset_name=args.dataset,
            input_parquet=args.input,
            output_json=args.output,
            use_judge=args.use_judge,
            cache_dir=args.cache_dir,
            limit=args.limit,
            workers=args.workers,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
