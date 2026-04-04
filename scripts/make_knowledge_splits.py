#!/usr/bin/env python3
"""
Split SimpleQA questions into knows/unknown based on pass@k correct counts.

  knows   — correct_count >= min_correct
  unknown — correct_count <  min_correct

Usage:
  python scripts/make_knowledge_splits.py \
    --eval-json outputs/simpleqa_qwen_pass_at_k/k_64/pass@k/eval_64.json \
    --source-data data/simpleqa/data.parquet \
    --output-dir data/simpleqa/splits \
    --min-correct 2
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True, help="eval_<k>.json from pass@k run")
    parser.add_argument("--source-data", default="data/simpleqa/data.parquet")
    parser.add_argument("--output-dir", default="data/simpleqa/splits")
    parser.add_argument("--min-correct", type=int, default=2,
                        help="Questions with >= this many correct rollouts go to knows")
    args = parser.parse_args()

    data = json.loads(Path(args.eval_json).read_text())
    rows = data["rows"]
    k = data["rows"][0]["graders"].__len__()  # infer k from grader count

    source_df = pd.read_parquet(args.source_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    knows_ids, unknown_ids = [], []
    for row in rows:
        n_correct = sum(1 for g in row["graders"] if g["evaluation"] == "correct")
        (knows_ids if n_correct >= args.min_correct else unknown_ids).append(row["id"])

    knows_df = source_df.iloc[knows_ids].copy().reset_index(drop=True)
    unknown_df = source_df.iloc[unknown_ids].copy().reset_index(drop=True)

    knows_df.to_parquet(output_dir / "knows.parquet", index=False)
    unknown_df.to_parquet(output_dir / "unknown.parquet", index=False)

    print(f"k={k}, min_correct={args.min_correct}")
    print(f"  knows:   {len(knows_df):5d} ({len(knows_df)/len(rows)*100:.1f}%)")
    print(f"  unknown: {len(unknown_df):5d} ({len(unknown_df)/len(rows)*100:.1f}%)")
    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    main()
