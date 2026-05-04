#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_decompose_mixed_k12_smoke_sft"

DATA_ROOT="data/simpleqa/partition/decompose_mixed_k12_smoke/sft"

TRAIN_DATA="$DATA_ROOT/train.parquet"

EVAL_DATA="$DATA_ROOT/train_eval_k1.parquet $DATA_ROOT/train_eval_k2.parquet $DATA_ROOT/train_eval_k2_unique.parquet $DATA_ROOT/train_eval_origqa.parquet"

# Pass@k disabled for now; re-enable by uncommenting the line below and the PASS_AT_K_EVAL_DATA env var.
# PASS_AT_K_EVAL_DATA="$DATA_ROOT/train_eval_origqa.parquet"

PROJECT_NAME="$PROJECT_NAME" \
TRAIN_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
PASS_AT_K_EVAL_DATA="" \
PROMPT_KEY="question" \
RESPONSE_KEY="answer" \
PROMPT_DICT_KEYS="" \
RESPONSE_DICT_KEYS="" \
MAX_LENGTH="4096" \
FILTER_OVERLONG_PROMPTS="1" \
LR_LIST="1.5e-4" \
EPOCHS_LIST="1 3 6 10 20 30" \
bash experiments/sft/run_sft_progressive_eval.sh
