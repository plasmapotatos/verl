#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_rich_qa_cot_sft"

DATA_DIR="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/cot/sft"

TRAIN_DATA="$DATA_DIR/train.parquet"
EVAL_DATA="$DATA_DIR/train_eval.parquet $DATA_DIR/val.parquet"
PASS_AT_K_EVAL_DATA="$DATA_DIR/val.parquet"

BASE_MODEL="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/richqa/sft_lr1.5e-4_epmax30_seed1/global_step_1290/merged_hf_model"

PROJECT_NAME="$PROJECT_NAME" \
BASE_MODEL="$BASE_MODEL" \
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
