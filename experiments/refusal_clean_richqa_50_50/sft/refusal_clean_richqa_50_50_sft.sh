#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_refusal_clean_richqa_50_50_sft"

DATA_DIR="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal_clean/sft/50-50"

TRAIN_DATA="$DATA_DIR/train_richqa_combined.parquet"

EVAL_DATA="$DATA_DIR/train_richqa_answer_frac0.2.parquet $DATA_DIR/train_richqa_answer_origqa_frac0.2.parquet $DATA_DIR/train_richqa_refusal_frac0.2.parquet $DATA_DIR/train_richqa_refusal_origqa_frac0.2.parquet"

PASS_AT_K_EVAL_DATA="$DATA_DIR/train_richqa_answer_origqa_frac0.2.parquet $DATA_DIR/train_richqa_refusal_origqa_frac0.2.parquet"

PROJECT_NAME="$PROJECT_NAME" \
TRAIN_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
PASS_AT_K_EVAL_DATA="$PASS_AT_K_EVAL_DATA" \
PROMPT_KEY="question" \
RESPONSE_KEY="answer" \
PROMPT_DICT_KEYS="" \
RESPONSE_DICT_KEYS="" \
MAX_LENGTH="4096" \
FILTER_OVERLONG_PROMPTS="1" \
LR_LIST="1.5e-4" \
EPOCHS_LIST="1 3 6 10 20 30" \
bash experiments/sft/run_sft_progressive_eval.sh
