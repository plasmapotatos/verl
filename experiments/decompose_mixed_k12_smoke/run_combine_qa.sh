#!/usr/bin/env bash
# Step 2: generate k=2 combined Q/A pairs for the selected ids.
# Run this with OPENAI_API_KEY set in the environment.
set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set" >&2
    exit 1
fi

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
DATA_DIR="$WORKDIR/data/simpleqa/partition/decompose_mixed_k12_smoke/_intermediate"
SIF="/work/hdd/bbsg/twei2/rl/torch2501.sif"

INPUT="$DATA_DIR/decompose_mid_subqas.parquet"
OUTPUT="$DATA_DIR/combined_k2.parquet"

# n=21 covers max C(7,2)=21 (we'll get 15 for g=6, 21 for g=7)
python -m verl.augment.cli combine_qa \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --k 2 \
    --n 21 \
    --model gpt-4o-mini \
    --batch_size 32 \
    --seed 0

echo "Wrote $OUTPUT"
