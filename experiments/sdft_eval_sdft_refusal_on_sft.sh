#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/work/hdd/bbsg/twei2/rl/Self-Distillation/outputs/sdft_refusal_on_sft_3b}"
RL_DIR="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/refusal/rl"
EVAL_DATA_TRAIN_EVAL_ANSWER="${EVAL_DATA_TRAIN_EVAL_ANSWER:-${RL_DIR}/train_eval_answer.parquet}"
EVAL_DATA_TRAIN_EVAL_REFUSAL="${EVAL_DATA_TRAIN_EVAL_REFUSAL:-${RL_DIR}/train_eval_refusal.parquet}"
EVAL_DATA_VAL_ANSWER="${EVAL_DATA_VAL_ANSWER:-${RL_DIR}/val_answer.parquet}"
EVAL_DATA_VAL_REFUSAL="${EVAL_DATA_VAL_REFUSAL:-${RL_DIR}/val_refusal.parquet}"
BASE_MODEL="${BASE_MODEL:-/work/hdd/bbsg/twei2/rl/verl/outputs/sft/simpleqa_rich_sft_refusal_partition_50_50/sft_lr1.5e-4_epmax30_seed1/global_step_1290/merged_hf_model}"

BASE_MODEL="$BASE_MODEL" \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/sdft_eval_and_plot.sh \
    "$OUTPUT_DIR" \
    "$EVAL_DATA_TRAIN_EVAL_ANSWER" \
    "$EVAL_DATA_TRAIN_EVAL_REFUSAL" \
    "$EVAL_DATA_VAL_ANSWER" \
    "$EVAL_DATA_VAL_REFUSAL"

[[ -n "${CHAIN_FLAG_FILE:-}" ]] && touch "$CHAIN_FLAG_FILE"
