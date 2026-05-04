"""Step 1: select 25% of ids whose decomposed group_size is in {6, 7}.

Outputs:
  _intermediate/selected_ids.txt        — one id per line.
  _intermediate/decompose_mid_subqas.parquet
                                        — train rows for selected ids only,
                                          input to combine_qa in step 2.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

DEFAULT_SRC = "/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose/sft/train.parquet"
OUT_DIR = Path("/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose_mixed_k12_smoke/_intermediate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--frac", type=float, default=0.25, help="fraction of total ids to keep")
    ap.add_argument("--allowed_sizes", type=int, nargs="+", default=[6, 7])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(args.src)
    sizes = df.groupby("id").size()
    total = len(sizes)
    target_n = int(round(args.frac * total))

    pool = sizes[sizes.isin(args.allowed_sizes)].index.tolist()
    if len(pool) < target_n:
        raise SystemExit(
            f"pool of size {len(pool)} (sizes={args.allowed_sizes}) < target {target_n}"
        )
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    selected = sorted(pool[:target_n], key=lambda x: int(x) if str(x).isdigit() else x)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids_path = OUT_DIR / "selected_ids.txt"
    ids_path.write_text("\n".join(str(i) for i in selected) + "\n")

    sub_df = df[df["id"].isin(selected)].reset_index(drop=True)
    sub_path = OUT_DIR / "decompose_mid_subqas.parquet"
    sub_df.to_parquet(sub_path, index=False)

    by_size = sub_df.groupby("id").size().value_counts().sort_index().to_dict()
    print(f"Selected {len(selected)} / {total} ids (target {target_n})")
    print(f"Group-size counts among selected: {by_size}")
    print(f"Wrote {ids_path}")
    print(f"Wrote {sub_path} ({len(sub_df)} rows)")


if __name__ == "__main__":
    main()
