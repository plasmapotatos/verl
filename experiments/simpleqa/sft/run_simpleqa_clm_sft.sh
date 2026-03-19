#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_clm"

TRAIN_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/clm/simpleqa_clm_train.parquet"

EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/clm/simpleqa_clm_train_frac0.1_origqa.parquet"

PASS_AT_K_EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/rich_sft/simpleqa_rich_sft_train_frac0.1_origqa.parquet"

PROJECT_NAME="$PROJECT_NAME" \
TRAIN_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
PASS_AT_K_EVAL_DATA="$PASS_AT_K_EVAL_DATA" \
CLM_MODE="1" \
CLM_TEXT_KEY="text" \
CLM_MAX_LEN="4096" \
CLM_TRUNCATION="right" \
EXP_PREFIX="clm" \
EPOCHS_LIST="1 3 6 10 20 30" \
LR_LIST="1.5e-4" \
bash experiments/sft/run_sft_sweep_with_eval.sh