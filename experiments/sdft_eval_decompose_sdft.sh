#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/work/hdd/bbsg/twei2/rl/Self-Distillation/outputs/sdft_decompose_sdft_3b}"
EVAL_DATA_1="${EVAL_DATA_1:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose/sft/train_eval.parquet}"
EVAL_DATA_2="${EVAL_DATA_2:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose/sft/train_eval_origqa.parquet}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

BASE_MODEL="$BASE_MODEL" \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/sdft_eval_and_plot.sh \
    "$OUTPUT_DIR" \
    "$EVAL_DATA_1" \
    "$EVAL_DATA_2"

[[ -n "${CHAIN_FLAG_FILE:-}" ]] && touch "$CHAIN_FLAG_FILE"
