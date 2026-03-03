#!/usr/bin/env bash
set -euo pipefail

VERL_DIR="${VERL_DIR:-/work/hdd/bbsg/twei2/rl/verl}"

PROJECT_NAME="${PROJECT_NAME:-simpleqa_clm}"
TRAIN_DATA="${TRAIN_DATA:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/simpleqa_clm_train.parquet}"
EVAL_DATA="${EVAL_DATA:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/base/train.parquet}"

CLM_MODE="${CLM_MODE:-1}"
CLM_TEXT_KEY="${CLM_TEXT_KEY:-text}"
CLM_MAX_LEN="${CLM_MAX_LEN:-4096}"
CLM_TRUNCATION="${CLM_TRUNCATION:-right}"

EXP_PREFIX="${EXP_PREFIX:-clm}"
EPOCHS_LIST="${EPOCHS_LIST:-1 3 6}"
LR_LIST="${LR_LIST:-1e-4}"

PROJECT_NAME="$PROJECT_NAME" \
TRAIN_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
CLM_MODE="$CLM_MODE" \
CLM_TEXT_KEY="$CLM_TEXT_KEY" \
CLM_MAX_LEN="$CLM_MAX_LEN" \
CLM_TRUNCATION="$CLM_TRUNCATION" \
EXP_PREFIX="$EXP_PREFIX" \
EPOCHS_LIST="$EPOCHS_LIST" \
LR_LIST="$LR_LIST" \
bash "$VERL_DIR/experiments/sft/run_sft_sweep_with_eval.sh"