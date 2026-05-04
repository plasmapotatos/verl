#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/work/hdd/bbsg/twei2/rl/Self-Distillation/outputs/sdft_refusal_from_base_qwen3b_3b}"
EVAL_DATA_ORIGQA_ANSWER="${EVAL_DATA_ORIGQA_ANSWER:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/refusal/sft/train_eval_origqa_answer.parquet}"
EVAL_DATA_ORIGQA_REFUSAL="${EVAL_DATA_ORIGQA_REFUSAL:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/refusal/sft/train_eval_origqa_refusal.parquet}"
EVAL_DATA_ANSWER="${EVAL_DATA_ANSWER:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/refusal/sft/train_eval_answer.parquet}"
EVAL_DATA_REFUSAL="${EVAL_DATA_REFUSAL:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/refusal/sft/train_eval_refusal.parquet}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

BASE_MODEL="$BASE_MODEL" \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/sdft_eval_and_plot.sh \
    "$OUTPUT_DIR" \
    "$EVAL_DATA_ORIGQA_ANSWER" \
    "$EVAL_DATA_ORIGQA_REFUSAL" \
    "$EVAL_DATA_ANSWER" \
    "$EVAL_DATA_REFUSAL"

[[ -n "${CHAIN_FLAG_FILE:-}" ]] && touch "$CHAIN_FLAG_FILE"
