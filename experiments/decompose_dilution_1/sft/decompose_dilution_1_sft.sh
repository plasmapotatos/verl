#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_decompose_dilution_1_sft"

TRAIN_DATA="data/simpleqa/partition/decompose_dilution_test/sft/1/train.parquet"

EVAL_DATA="data/simpleqa/partition/decompose_dilution_test/sft/1/train_eval.parquet data/simpleqa/partition/decompose_dilution_test/sft/1/train_eval_origqa.parquet"

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
bash experiments/sft/run_sft_progressive_eval.sh
