#!/usr/bin/env bash
# Prepare factual_anchor data splits following the rich_sft naming convention.
#
# Steps:
#   1. Filter both datasets to their ID intersection (overwrites base files)
#   2. Split factual_anchor 90-10 via make_eval_subset.py
#   3. Sample 10% of the 90% train for pass@k eval
#   4. Align origqa splits to match factual_anchor splits by ID
set -euo pipefail

DIR="data/simpleqa/augment/factual_anchor"

FA="$DIR/simpleqa_rich_sft_train_factual_anchor.parquet"
OQ="$DIR/simpleqa_rich_sft_train_origqa_factual_anchor.parquet"

FA_EVAL="$DIR/simpleqa_rich_sft_train_factual_anchor_frac0.1.parquet"
FA_TRAIN="$DIR/simpleqa_rich_sft_train_factual_anchor_frac0.9_train.parquet"
FA_PASSK="$DIR/simpleqa_rich_sft_train_factual_anchor_frac0.9_train_frac0.1.parquet"

PYTHON="${PYTHON:-python}"

echo "=== Step 1: Filter to ID intersection ==="
"$PYTHON" scripts/prepare_factual_anchor_splits.py filter \
    --factual-anchor "$FA" \
    --origqa "$OQ"

echo ""
echo "=== Step 2: Split factual_anchor 90-10 ==="
"$PYTHON" scripts/make_eval_subset.py \
    --input "$FA" \
    --output "$FA_EVAL" \
    --fraction 0.1 \
    --seed 1 \
    --save-remainder \
    --train-output "$FA_TRAIN"

echo ""
echo "=== Step 3: Sample 10% of train for pass@k eval ==="
"$PYTHON" scripts/make_eval_subset.py \
    --input "$FA_TRAIN" \
    --output "$FA_PASSK" \
    --fraction 0.1 \
    --seed 1

echo ""
echo "=== Step 4: Align origqa splits to factual_anchor splits ==="
"$PYTHON" scripts/prepare_factual_anchor_splits.py align-origqa \
    --origqa "$OQ" \
    --fa-eval "$FA_EVAL" \
    --fa-train "$FA_TRAIN" \
    --fa-passk "$FA_PASSK" \
    --out-dir "$DIR"

echo ""
echo "=== Done. Output files ==="
ls -lh "$DIR"/simpleqa_rich_sft_train_factual_anchor*.parquet
ls -lh "$DIR"/simpleqa_rich_sft_train_origqa_factual_anchor*.parquet
