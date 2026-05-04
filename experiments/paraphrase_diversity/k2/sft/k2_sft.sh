#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_paraphrase_diversity_k2_sft"

TRAIN_DATA="data/simpleqa/partition/paraphrase_both/train_k2.parquet"

EVAL_DATA="data/simpleqa/partition/paraphrase_both/train_eval.parquet data/simpleqa/partition/paraphrase_both/train_eval_origqa.parquet"

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
EPOCHS_LIST="1 3 6 10 20 30" \
bash experiments/sft/run_sft_progressive_eval.sh
