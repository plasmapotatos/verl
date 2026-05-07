#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_refusal_ethan_ternary_compare_sft"

DATA_DIR="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/refusal_ethan/sft"

TRAIN_DATA="$DATA_DIR/train.parquet"

EVAL_DATA="$DATA_DIR/answer_eval.parquet $DATA_DIR/answer_eval_origqa.parquet $DATA_DIR/refuse_eval.parquet $DATA_DIR/refuse_eval_origqa.parquet"

PROJECT_NAME="$PROJECT_NAME" \
TRAIN_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
PROMPT_KEY="question" \
RESPONSE_KEY="answer" \
PROMPT_DICT_KEYS="" \
RESPONSE_DICT_KEYS="" \
MAX_LENGTH="4096" \
FILTER_OVERLONG_PROMPTS="1" \
LR_LIST="1.5e-4" \
EPOCHS_LIST="1 3 6 10 20" \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/sft/run_sft_progressive_eval.sh
