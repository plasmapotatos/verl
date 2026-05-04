#!/usr/bin/env bash
# Build eval companions for rl_k3:
#   train_eval.parquet     10% row subsample of rl_k3/train.parquet
#   train_eval_k4.parquet  k=4 combos for the facts in train_eval (1 per fact)
#   val_k4.parquet         k=4 combos for the facts in rl_k3/val.parquet (1 per fact)
#
# Tunables (env):
#   TRAIN_EVAL_FRAC  fraction of rl_k3/train.parquet rows to keep (default 0.10)
#   SEED             (default 0)
#
# Requires OPENAI_API_KEY (for k=4 combine_qa, if combined_k4_eval.parquet
# isn't already cached).
set -euo pipefail

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
SIF="/work/hdd/bbsg/twei2/rl/torch2501.sif"
ROOT="$WORKDIR/data/simpleqa/partition/decompose_mixed_k12_smoke"
INTER="$ROOT/_intermediate"
SFT="$ROOT/sft"
RL_K3="$ROOT/rl_k3"

TRAIN_EVAL_FRAC="${TRAIN_EVAL_FRAC:-0.10}"
SEED="${SEED:-0}"

FILTERED_INPUT="$INTER/k4_eval_input.parquet"
COMBINED_K4="$INTER/combined_k4_eval.parquet"

mkdir -p "$INTER" "$RL_K3"

# Step 1: subsample train_eval and dump filtered input for combine_qa.
python - <<PY
from pathlib import Path
import numpy as np
import pandas as pd

RL_K3 = Path("$RL_K3")
SFT = Path("$SFT")
FILTERED = Path("$FILTERED_INPUT")
frac = float("$TRAIN_EVAL_FRAC")
seed = int("$SEED")

train = pd.read_parquet(RL_K3 / "train.parquet")
val = pd.read_parquet(RL_K3 / "val.parquet")

n = max(1, int(round(len(train) * frac)))
train_eval = train.sample(n=n, random_state=seed).reset_index(drop=True)
train_eval.to_parquet(RL_K3 / "train_eval.parquet", index=False)
print(f"train_eval: {len(train_eval)} rows / {train_eval['id'].nunique()} facts")

# Filter sft/train.parquet to the union of fact ids needed for k=4 builds.
needed_ids = sorted(set(train_eval["id"]).union(val["id"]))
sft_train = pd.read_parquet(SFT / "train.parquet")
filt = sft_train[sft_train["id"].isin(needed_ids)].reset_index(drop=True)
filt.to_parquet(FILTERED, index=False)
print(f"filtered sft input: {len(filt)} rows / {filt['id'].nunique()} facts -> {FILTERED}")
PY

# Step 2: combine_qa at k=4 on the filtered input.
if [[ -f "$COMBINED_K4" ]]; then
    echo "[combine_qa k=4] $COMBINED_K4 exists, skipping"
else
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "ERROR: OPENAI_API_KEY is not set" >&2
        exit 1
    fi
    echo "[combine_qa k=4] -> $COMBINED_K4"
    python -m verl.augment.cli combine_qa \
        --input "$FILTERED_INPUT" \
        --output "$COMBINED_K4" \
        --k 4 \
        --n 1 \
        --model gpt-4o-mini \
        --batch_size 32 \
        --seed "$SEED"
fi

# Step 3: wrap and split into train_eval_k4 / val_k4.
python - <<PY
from pathlib import Path
import numpy as np
import pandas as pd

RL_K3 = Path("$RL_K3")
SFT = Path("$SFT")
COMBINED = Path("$COMBINED_K4")

train_eval = pd.read_parquet(RL_K3 / "train_eval.parquet")
val = pd.read_parquet(RL_K3 / "val.parquet")
sft_train = pd.read_parquet(SFT / "train.parquet")
templates = sft_train.drop_duplicates("id").set_index("id")

train_eval_ids = set(train_eval["id"])
val_ids = set(val["id"])

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
            "k": 4,
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

combined = (
    pd.read_parquet(COMBINED)
      .sort_values(["id", "draw_idx"])
      .groupby("id", as_index=False)
      .head(1)
      .reset_index(drop=True)
)

rows = []
for i, row in combined.iterrows():
    gid = row["id"]
    if gid not in templates.index:
        print(f"  skip id={gid}: no template row")
        continue
    rows.append(build_row(row.to_dict(), templates.loc[gid], i))
df = pd.DataFrame(rows)

train_eval_k4 = df[df["id"].isin(train_eval_ids)].reset_index(drop=True)
val_k4 = df[df["id"].isin(val_ids)].reset_index(drop=True)

train_eval_k4.to_parquet(RL_K3 / "train_eval_k4.parquet", index=False)
val_k4.to_parquet(RL_K3 / "val_k4.parquet", index=False)
print(f"train_eval_k4: {len(train_eval_k4)} rows / {train_eval_k4['id'].nunique()} facts")
print(f"val_k4:        {len(val_k4)} rows / {val_k4['id'].nunique()} facts")
PY

echo "Done."
