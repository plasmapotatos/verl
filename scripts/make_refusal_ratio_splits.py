#!/usr/bin/env python3
"""
Create 75-25 and 25-75 answer/refusal ratio splits for SFT and RL data.

First number = proportion of answer (factual) targets.
Second number = proportion of IDK/refusal targets.

50-50 data already exists; this creates 75-25 and 25-75 splits.

Usage:
    python scripts/make_refusal_ratio_splits.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

SEED = 42
SFT_50_50 = Path("/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal/sft/50-50")
SFT_OUT = Path("/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal/sft")
RL_DIR = Path("/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal/rl")
RL_OUT = Path("/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal/rl")

RATIOS = [
    ("75-25", 0.75, 0.25),
    ("25-75", 0.25, 0.75),
]


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  Wrote {len(df):>5} rows → {path}")


def frac_sample(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    n = max(1, round(len(df) * frac))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def make_sft_split(name: str, ans_frac: float, ref_frac: float) -> None:
    """Create SFT split directory with all required files."""
    print(f"\n=== SFT {name} ===")
    out = SFT_OUT / name

    # Load full 50-50 data
    df_ans = pd.read_parquet(SFT_50_50 / "train_richqa_answer.parquet")
    df_ref = pd.read_parquet(SFT_50_50 / "train_richqa_refusal.parquet")
    df_ans_origqa = pd.read_parquet(SFT_50_50 / "train_richqa_answer_origqa.parquet")
    df_ref_origqa = pd.read_parquet(SFT_50_50 / "train_richqa_refusal_origqa.parquet")

    # For 75-25: use all answer, subsample refusal to match ratio
    # For 25-75: subsample answer, use all refusal
    if ans_frac >= ref_frac:
        # answer is dominant: use all answer, subsample refusal
        ratio = ref_frac / ans_frac
        df_ans_split = df_ans.reset_index(drop=True)
        df_ref_split = df_ref.sample(n=round(len(df_ans) * ratio), random_state=SEED).reset_index(drop=True)
    else:
        # refusal is dominant: subsample answer, use all refusal
        ratio = ans_frac / ref_frac
        df_ans_split = df_ans.sample(n=round(len(df_ref) * ratio), random_state=SEED).reset_index(drop=True)
        df_ref_split = df_ref.reset_index(drop=True)

    print(f"  answer: {len(df_ans_split)}, refusal: {len(df_ref_split)}, total: {len(df_ans_split)+len(df_ref_split)}")

    # Align origqa files to sampled IDs
    ans_ids = set(df_ans_split["id"].astype(str))
    ref_ids = set(df_ref_split["id"].astype(str))
    df_ans_origqa_split = df_ans_origqa[df_ans_origqa["id"].astype(str).isin(ans_ids)].reset_index(drop=True)
    df_ref_origqa_split = df_ref_origqa[df_ref_origqa["id"].astype(str).isin(ref_ids)].reset_index(drop=True)

    # Save training files
    save(df_ans_split, out / "train_richqa_answer.parquet")
    save(df_ref_split, out / "train_richqa_refusal.parquet")
    save(pd.concat([df_ans_split, df_ref_split], ignore_index=True), out / "train_richqa_combined.parquet")

    # Save origqa files (for eval)
    save(df_ans_origqa_split, out / "train_richqa_answer_origqa.parquet")
    save(df_ref_origqa_split, out / "train_richqa_refusal_origqa.parquet")

    # Save frac0.2 eval subsets (20% sample of each)
    save(frac_sample(df_ans_split, 0.2, SEED), out / "train_richqa_answer_frac0.2.parquet")
    save(frac_sample(df_ref_split, 0.2, SEED), out / "train_richqa_refusal_frac0.2.parquet")
    save(frac_sample(df_ans_origqa_split, 0.2, SEED), out / "train_richqa_answer_origqa_frac0.2.parquet")
    save(frac_sample(df_ref_origqa_split, 0.2, SEED), out / "train_richqa_refusal_origqa_frac0.2.parquet")

    # Write metadata
    (out / "metadata.txt").write_text(
        f"1st number is proportion of answer target, second is IDK target\n"
        f"Split: {name}\n"
        f"Answer rows: {len(df_ans_split)} (of {len(df_ans)} available)\n"
        f"Refusal rows: {len(df_ref_split)} (of {len(df_ref)} available)\n"
        f"Total training: {len(df_ans_split)+len(df_ref_split)}\n"
        f"Seed: {SEED}\n"
    )
    print(f"  Wrote metadata → {out / 'metadata.txt'}")


def make_rl_split(name: str, ans_frac: float, ref_frac: float) -> None:
    """Create RL split directory with all required files."""
    print(f"\n=== RL {name} ===")
    out = RL_OUT / name

    # Split existing RL combined by augmentation method
    df_comb = pd.read_parquet(RL_DIR / "train_richqa_rl_combined.parquet")
    df_ans = df_comb[
        df_comb["extra_info"].apply(lambda x: x["augmentation"]["method"]) == "simpleqa_rich_sft"
    ].reset_index(drop=True)
    df_ref = df_comb[
        df_comb["extra_info"].apply(lambda x: x["augmentation"]["method"]) == "simpleqa_refusal"
    ].reset_index(drop=True)

    # Load existing origqa eval files (have correct RL format with prompt/reward_model)
    df_ans_origqa_full = pd.read_parquet(RL_DIR / "train_richqa_answer_origqa_frac0.8.parquet")
    df_ref_origqa_full = pd.read_parquet(RL_DIR / "train_richqa_refusal_origqa_frac0.8.parquet")
    df_ans_origqa_holdout = pd.read_parquet(RL_DIR / "train_richqa_answer_origqa_frac0.2.parquet")
    df_ref_origqa_holdout = pd.read_parquet(RL_DIR / "train_richqa_refusal_origqa_frac0.2.parquet")

    # Subsample to target ratio (same logic as SFT)
    if ans_frac >= ref_frac:
        ratio = ref_frac / ans_frac
        df_ans_split = df_ans.reset_index(drop=True)
        df_ref_split = df_ref.sample(n=round(len(df_ans) * ratio), random_state=SEED).reset_index(drop=True)
    else:
        ratio = ans_frac / ref_frac
        df_ans_split = df_ans.sample(n=round(len(df_ref) * ratio), random_state=SEED).reset_index(drop=True)
        df_ref_split = df_ref.reset_index(drop=True)

    print(f"  answer: {len(df_ans_split)}, refusal: {len(df_ref_split)}, total: {len(df_ans_split)+len(df_ref_split)}")

    ans_ids = set(df_ans_split["id"].astype(str))
    ref_ids = set(df_ref_split["id"].astype(str))

    # Training combined (with target column)
    df_combined = pd.concat([df_ans_split, df_ref_split], ignore_index=True)
    save(df_combined, out / "train_richqa_rl_combined.parquet")

    # origqa frac0.8: filter full origqa files to sampled IDs
    df_ans_origqa_split = df_ans_origqa_full[
        df_ans_origqa_full["id"].astype(str).isin(ans_ids)
    ].reset_index(drop=True)
    df_ref_origqa_split = df_ref_origqa_full[
        df_ref_origqa_full["id"].astype(str).isin(ref_ids)
    ].reset_index(drop=True)
    save(df_ans_origqa_split, out / "train_richqa_answer_origqa_frac0.8.parquet")
    save(df_ref_origqa_split, out / "train_richqa_refusal_origqa_frac0.8.parquet")

    # origqa frac0.8_frac0.2: 20% of the training origqa for in-distribution eval
    save(frac_sample(df_ans_origqa_split, 0.2, SEED), out / "train_richqa_answer_origqa_frac0.8_frac0.2.parquet")
    save(frac_sample(df_ref_origqa_split, 0.2, SEED), out / "train_richqa_refusal_origqa_frac0.8_frac0.2.parquet")

    # origqa frac0.2 holdout: rows NOT in training set (out-of-distribution eval)
    # For dominant class (all rows used): use the existing holdout file
    # For minority class (subsampled): holdout = remaining rows not selected for training
    if ans_frac >= ref_frac:
        # answer uses all available: use existing holdout
        save(df_ans_origqa_holdout, out / "train_richqa_answer_origqa_frac0.2.parquet")
        # refusal: holdout = rows in origqa_full not in training set
        ref_holdout_ids = set(df_ref_origqa_full["id"].astype(str)) - ref_ids
        df_ref_holdout = df_ref_origqa_full[
            df_ref_origqa_full["id"].astype(str).isin(ref_holdout_ids)
        ].reset_index(drop=True)
        save(df_ref_holdout, out / "train_richqa_refusal_origqa_frac0.2.parquet")
    else:
        # refusal uses all available: use existing holdout
        save(df_ref_origqa_holdout, out / "train_richqa_refusal_origqa_frac0.2.parquet")
        # answer: holdout = rows in origqa_full not in training set
        ans_holdout_ids = set(df_ans_origqa_full["id"].astype(str)) - ans_ids
        df_ans_holdout = df_ans_origqa_full[
            df_ans_origqa_full["id"].astype(str).isin(ans_holdout_ids)
        ].reset_index(drop=True)
        save(df_ans_holdout, out / "train_richqa_answer_origqa_frac0.2.parquet")

    # Write metadata
    (out / "metadata.txt").write_text(
        f"1st number is proportion of answer target, second is IDK target\n"
        f"Split: {name}\n"
        f"Answer rows: {len(df_ans_split)} (of {len(df_ans)} available)\n"
        f"Refusal rows: {len(df_ref_split)} (of {len(df_ref)} available)\n"
        f"Total training: {len(df_ans_split)+len(df_ref_split)}\n"
        f"Seed: {SEED}\n"
    )
    print(f"  Wrote metadata → {out / 'metadata.txt'}")


def main() -> None:
    for name, ans_frac, ref_frac in RATIOS:
        make_sft_split(name, ans_frac, ref_frac)
        make_rl_split(name, ans_frac, ref_frac)
    print("\nDone.")


if __name__ == "__main__":
    main()
