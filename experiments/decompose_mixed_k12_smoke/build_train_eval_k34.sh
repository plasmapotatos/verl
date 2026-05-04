#!/usr/bin/env bash
# Run the composition augmentor (combine_qa) at k=3 and k=4 on the SFT
# train split, then wrap each output into the eval-ready parquet format
# (one row per id, with `prompt` / `reward_model` / `extra_info`) so they
# can be consumed by verl.trainer.main_generation.
#
# Inputs:
#   data/.../decompose_mixed_k12_smoke/sft/train.parquet
#
# Outputs (under sft/):
#   train_eval_k3.parquet
#   train_eval_k4.parquet
#
# Intermediate (under _intermediate/):
#   combined_k3.parquet
#   combined_k4.parquet
#
# Requires OPENAI_API_KEY.

set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set" >&2
    exit 1
fi

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
ROOT="$WORKDIR/data/simpleqa/partition/decompose_mixed_k12_smoke"
INTER="$ROOT/_intermediate"
SFT="$ROOT/sft"
SIF="/work/hdd/bbsg/twei2/rl/torch2501.sif"

INPUT="$SFT/train.parquet"
mkdir -p "$INTER" "$SFT"

run_combine() {
    local k="$1"
    local n="$2"
    local out="$INTER/combined_k${k}.parquet"
if [[ -f "$out" ]]; then
        echo "[k=$k] $out exists, skipping combine_qa"
        return 0
    fi
    echo "[k=$k] running combine_qa -> $out"
    python -m verl.augment.cli combine_qa \
        --input "$INPUT" \
        --output "$out" \
        --k "$k" \
        --n "$n" \
        --model gpt-4o-mini \
        --batch_size 32 \
        --seed 0
}

# n=1 draw per id is enough to produce one eval row per id.
run_combine 3 1
run_combine 4 1

# Wrap combined_kN.parquet into eval-ready train_eval_kN.parquet.
python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose_mixed_k12_smoke")
SFT = ROOT / "sft"
INTER = ROOT / "_intermediate"

train = pd.read_parquet(SFT / "train.parquet")
templates = train.drop_duplicates("id").set_index("id")

def normalize_metadata(md):
    if isinstance(md, dict):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in md.items()}
    return md

def build_row(combined: dict, template: pd.Series, idx: int, k: int) -> dict:
    gid = combined["id"]
    q = combined["question"]
    a = combined["answer"]
    metadata = normalize_metadata(template.get("metadata"))
    extra_info = {
        "answer": a,
        "augmentation": {
            "method": "combine_qa",
            "original_sample_id": gid,
            "draw_idx": int(combined["draw_idx"]),
            "k": k,
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
        "ability": "general",
        "reward_model": {"ground_truth": a, "style": "rule"},
        "extra_info": extra_info,
    }

for k in (3, 4):
    src = INTER / f"combined_k{k}.parquet"
    dst = SFT / f"train_eval_k{k}.parquet"
    combined = pd.read_parquet(src).sort_values(["id", "draw_idx"])
    combined = combined.groupby("id", as_index=False).head(1).reset_index(drop=True)
    rows = []
    for i, row in combined.iterrows():
        gid = row["id"]
        if gid not in templates.index:
            print(f"  [k={k}] skip id={gid}: no template row in train.parquet")
            continue
        rows.append(build_row(row.to_dict(), templates.loc[gid], i, k))
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(dst, index=False)
    print(f"  [k={k}] wrote {len(out_df)} rows -> {dst}")
PY

echo "Done."
