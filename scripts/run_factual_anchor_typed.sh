#!/usr/bin/env bash
# Type bracketed factual anchors in all smoketest parquets.
# Input:  data/.../unbracketed/{sft,rl}/*.parquet (bracketed, untouched)
# Output: data/.../unbracketed/typed/{sft,rl}/*.parquet (overwritten)
set -euo pipefail

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be exported before running}"

SIF="/work/hdd/bbsg/twei2/rl/torch2501.sif"
BASE="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/factual_anchor/smoketest/unbracketed"
BATCH_SIZE="${BATCH_SIZE:-64}"

FILES=(
    "sft/train.parquet"
    "sft/train_eval.parquet"
    "sft/train_origqa.parquet"
    "sft/train_eval_origqa.parquet"
    "rl/train.parquet"
    "rl/train_eval.parquet"
    "rl/val.parquet"
)

for rel in "${FILES[@]}"; do
    in="$BASE/$rel"
    out="$BASE/typed/$rel"
    echo "=== $rel ==="
    mkdir -p "$(dirname "$out")"
    python -m verl.augment.cli \
        --input "$in" \
        --output "$out" \
        --method factual_anchor_typed \
        --batch_size "$BATCH_SIZE"
done

echo "Done. Typed outputs written under $BASE/typed/"
