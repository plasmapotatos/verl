"""Command-line interface for dataset augmentation."""

from __future__ import annotations

import argparse
import sys

from .registry import list_methods
from .runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VERL dataset augmentation")
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--output", help="Output parquet path")
    parser.add_argument("--output_dir", help="Output directory for per-method files")
    parser.add_argument("--method", action="append", default=[], help="Method name (repeatable)")
    parser.add_argument("--mode", help="Mode for LLM rewriter prompts")
    parser.add_argument("--n", type=int, default=1, help="Variants per method")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed")
    parser.add_argument("--write_per_method", action="store_true", help="Write per-method outputs")
    parser.add_argument("--mix_original", action="store_true", help="Include original samples")
    parser.add_argument("--max_samples", type=int, help="Max samples for debugging")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of samples to process concurrently",
    )
    parser.add_argument("--list_methods", action="store_true", help="List methods and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_methods:
        for name in list_methods():
            print(name)
        return 0

    if not args.method:
        parser.error("--method is required unless --list_methods is set")

    if args.write_per_method and not args.output_dir:
        parser.error("--output_dir is required when --write_per_method is set")

    if not args.write_per_method and not args.output:
        parser.error("--output is required when --write_per_method is not set")

    run(
        input_path=args.input,
        output_path=args.output,
        output_dir=args.output_dir,
        methods=args.method,
        n_variants_per_method=args.n,
        seed=args.seed,
        write_per_method=args.write_per_method,
        mix_original=args.mix_original,
        mode=args.mode,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
