#!/usr/bin/env bash
# Split sft/train_eval_k3.parquet into rl/train.parquet and rl/val.parquet
# on disjoint fact ids. The k=3 eval set was built from the same 688 facts
# used in sft/train.parquet (see build_train_eval_k34.sh), so this just
# partitions those facts for RL.
#
# Outputs (under data/.../decompose_mixed_k12_smoke/rl/):
#   train.parquet
#   val.parquet
#
# Tunables:
#   VAL_SIZE  (default: 150)  number of facts held out for RL val
#   SEED      (default: 0)
set -euo pipefail

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
SIF="/work/hdd/bbsg/twei2/rl/torch2501.sif"
ROOT="$WORKDIR/data/simpleqa/partition/decompose_mixed_k12_smoke"
SRC="$ROOT/sft/train_eval_k3.parquet"
OUT="$ROOT/rl_k3"
mkdir -p "$OUT"

VAL_SIZE="${VAL_SIZE:-150}"
SEED="${SEED:-0}"

python - <<PY
import pandas as pd, numpy as np
from pathlib import Path

src = Path("$SRC")
out = Path("$OUT")
val_size = int("$VAL_SIZE")
seed = int("$SEED")

df = pd.read_parquet(src)
assert df["id"].is_unique, "expected one row per fact id"
n = len(df)
assert val_size < n, f"val_size={val_size} >= n={n}"

rng = np.random.default_rng(seed)
perm = rng.permutation(n)
val_idx = sorted(perm[:val_size].tolist())
train_idx = sorted(perm[val_size:].tolist())

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df   = df.iloc[val_idx].reset_index(drop=True)

train_df.to_parquet(out / "train.parquet", index=False)
val_df.to_parquet(out / "val.parquet", index=False)

print(f"wrote {len(train_df)} train, {len(val_df)} val (of {n}) -> {out}")
print(f"  train ids[:5]: {train_df['id'].head().tolist()}")
print(f"  val   ids[:5]: {val_df['id'].head().tolist()}")
PY
