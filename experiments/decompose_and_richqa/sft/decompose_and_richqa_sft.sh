#!/usr/bin/env bash
# SFT on decomposed sub-QA (with SimpleQA-like dupes dropped) combined with rich_qa.
set -euo pipefail

PROJECT_NAME="simpleqa_decompose_and_richqa_sft"

TRAIN_DATA="data/simpleqa/partition/decompose_and_richqa/sft/train.parquet"

# train_eval.parquet: full combined eval
# train_eval_origqa.parquet: the original questions covered by these samples
EVAL_DATA="data/simpleqa/partition/decompose_and_richqa/sft/train_eval.parquet data/simpleqa/partition/decompose_and_richqa/sft/train_eval_origqa.parquet"

PASS_AT_K_EVAL_DATA="data/simpleqa/partition/decompose_and_richqa/sft/train_eval_origqa.parquet"

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
