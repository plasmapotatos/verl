#!/usr/bin/env bash
# Build a k=3 RL train/val pair for the decompose_mixed_k12_smoke experiment.
#
# 1. combine_qa --k 3 --n N  on sft/train.parquet (688 unique facts, 6-7 subqs each).
# 2. Wrap into eval-ready rows (prompt / reward_model / extra_info).
# 3. Split by fact id (disjoint facts) into rl_k3/train.parquet and rl_k3/val.parquet.
#    - Train facts keep all N combos.
#    - Val facts keep 1 combo (lean held-out eval).
#
# Tunables (env):
#   N          combos per fact (default 4)
#   VAL_FACTS  number of facts held out for val (default 138 ≈ 80/20)
#   SEED       (default 0)
#
# Requires OPENAI_API_KEY.
set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set" >&2
    exit 1
fi

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
SIF="/work/hdd/bbsg/twei2/rl/torch2501.sif"
ROOT="$WORKDIR/data/simpleqa/partition/decompose_mixed_k12_smoke"
INTER="$ROOT/_intermediate"
SFT="$ROOT/sft"
OUT="$ROOT/rl_k3"

N="${N:-4}"
VAL_FACTS="${VAL_FACTS:-138}"
SEED="${SEED:-0}"

INPUT="$SFT/train.parquet"
COMBINED="$INTER/combined_k3_n${N}.parquet"
mkdir -p "$INTER" "$OUT"

if [[ -f "$COMBINED" ]]; then
    echo "[combine_qa] $COMBINED exists, skipping"
else
    echo "[combine_qa] N=$N -> $COMBINED"
    python -m verl.augment.cli combine_qa \
        --input "$INPUT" \
        --output "$COMBINED" \
        --k 3 \
        --n "$N" \
        --model gpt-4o-mini \
        --batch_size 32 \
        --seed "$SEED"
fi

python - <<PY
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("$ROOT")
SFT = Path("$SFT")
COMBINED = Path("$COMBINED")
OUT = Path("$OUT")
VAL_FACTS = int("$VAL_FACTS")
SEED = int("$SEED")

train = pd.read_parquet(SFT / "train.parquet")
templates = train.drop_duplicates("id").set_index("id")

def normalize_metadata(md):
    if isinstance(md, dict):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in md.items()}
    return md

def build_row(combined: dict, template: pd.Series, idx: int) -> dict:
    gid = combined["id"]
    q = combined["question"]
    a = combined["answer"]
    metadata = normalize_metadata(template.get("metadata"))
    extra_info = {
        "answer": a,
        "facts": list(combined["source_answers"]),
        "augmentation": {
            "method": "combine_qa",
            "original_sample_id": gid,
            "draw_idx": int(combined["draw_idx"]),
            "k": 3,
            "source_subq_indices": list(combined["source_subq_indices"]),
            "source_questions": list(combined["source_questions"]),
            "source_answers": list(combined["source_answers"]),
        },
        "index": idx,
        "metadata": metadata,
        "original_answer": a,
        "original_question": q,
        "question": q,
        "sample_id": gid,
        "split": "train",
    }
    return {
        "id": gid,
        "question": q,
        "answer": a,
        "metadata": metadata,
        "data_source": "simpleqa",
        "prompt": [{"content": q, "role": "user"}],
        "ability": template.get("ability", "general"),
        "reward_model": {"ground_truth": a, "style": "rule"},
        "extra_info": extra_info,
    }

combined = pd.read_parquet(COMBINED).sort_values(["id", "draw_idx"]).reset_index(drop=True)
rows = []
for i, row in combined.iterrows():
    gid = row["id"]
    if gid not in templates.index:
        print(f"  skip id={gid}: no template row")
        continue
    rows.append(build_row(row.to_dict(), templates.loc[gid], i))
df = pd.DataFrame(rows)
print(f"built {len(df)} rows over {df['id'].nunique()} facts")

# Disjoint-fact split.
all_ids = sorted(df["id"].unique().tolist())
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(all_ids))
val_ids = set(all_ids[i] for i in perm[:VAL_FACTS])
train_ids = set(all_ids[i] for i in perm[VAL_FACTS:])
assert val_ids.isdisjoint(train_ids)

train_df = df[df["id"].isin(train_ids)].reset_index(drop=True)
# Val: 1 combo per fact (smallest draw_idx; df is already sorted by id, draw_idx).
val_df = (
    df[df["id"].isin(val_ids)]
      .groupby("id", as_index=False)
      .head(1)
      .reset_index(drop=True)
)

train_df.to_parquet(OUT / "train.parquet", index=False)
val_df.to_parquet(OUT / "val.parquet", index=False)
print(f"wrote train: {len(train_df)} rows / {train_df['id'].nunique()} facts -> {OUT/'train.parquet'}")
print(f"wrote val:   {len(val_df)} rows / {val_df['id'].nunique()} facts -> {OUT/'val.parquet'}")
PY

echo "Done."
