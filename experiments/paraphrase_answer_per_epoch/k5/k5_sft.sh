#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_paraphrase_answer_per_epoch_k5_sft"

DATA_DIR="data/simpleqa/partition/paraphrase_answer"
TRAIN_DATA="$DATA_DIR/train_k5.parquet"
EVAL_DATA="$DATA_DIR/train_eval.parquet $DATA_DIR/train_eval_origqa.parquet"

# k5: 13760 rows / bs 64 = 215 steps/epoch -> save every epoch
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
EPOCHS_LIST="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20" \
SAVE_FREQ="215" \
bash experiments/sft/run_sft_progressive_eval.sh
