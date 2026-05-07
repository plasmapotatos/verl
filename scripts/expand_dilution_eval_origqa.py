"""Expand train_eval_origqa.parquet for the decompose-dilution sweep from
550 ids to 1500 ids (keeping the existing 550 as a strict subset).

Source pool: partition/rich_qa/sft/train_origqa.parquet (2752 rows, one per id,
schema matches the existing eval split). The same 1500-row file is written
into every dilution dir (sft/{0, 0_25, 0_5, 0_75, 1}).

A backup of the original 550-row file is saved alongside as
train_eval_origqa_550.parquet.
"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

SEED = 17
NEW_SIZE = 1500

BASE = Path("/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose_dilution_test/sft")
SRC = Path("/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/sft/train_origqa.parquet")
DILUTIONS = ["0", "0_25", "0_5", "0_75", "1"]

src = pd.read_parquet(SRC)
existing = pd.read_parquet(BASE / "0" / "train_eval_origqa.parquet")
existing_ids = list(existing["id"])
assert set(existing_ids).issubset(set(src["id"])), "existing eval ids not all in source"

remaining = src[~src["id"].isin(existing_ids)].reset_index(drop=True)
n_extra = NEW_SIZE - len(existing_ids)
assert n_extra > 0, f"NEW_SIZE {NEW_SIZE} <= existing {len(existing_ids)}"
assert n_extra <= len(remaining), f"asked for {n_extra} extras, only {len(remaining)} available"

rng = np.random.default_rng(SEED)
extra_idx = rng.choice(len(remaining), size=n_extra, replace=False)
extra = remaining.iloc[extra_idx].reset_index(drop=True)

# preserve existing rows verbatim, then append extras
combined = pd.concat([existing, extra], ignore_index=True)
assert combined["id"].nunique() == NEW_SIZE
assert set(existing_ids).issubset(set(combined["id"]))

print(f"existing: {len(existing)}, extras: {len(extra)}, combined: {len(combined)}")

for tag in DILUTIONS:
    out_dir = BASE / tag
    target = out_dir / "train_eval_origqa.parquet"
    backup = out_dir / "train_eval_origqa_550.parquet"
    if target.exists() and not backup.exists():
        shutil.copy2(target, backup)
        print(f"[{tag}] backed up -> {backup.name}")
    combined.to_parquet(target, index=False)
    print(f"[{tag}] wrote {target} ({len(combined)} rows)")

print("done.")
