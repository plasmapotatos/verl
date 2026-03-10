#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_clm_grpo"
EXPERIMENT_NAME="simpleqa_clm_grpo"

MODEL_PATH="/work/hdd/bbsg/twei2/rl/verl/outputs/simpleqa_clm/clm_lr1.5e-4_ep10_seed1/global_step_320/merged_hf_model"
TRAIN_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/clm/simpleqa_clm_train_origqa.parquet"
EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/clm/simpleqa_clm_train_origqa.parquet"

PROJECT_NAME="$PROJECT_NAME" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
MODEL_PATH="$MODEL_PATH" \
TRAIN_DATA="$TRAIN_DATA" \
VAL_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
TOTAL_EPOCHS=60 \
RUN_EVAL=1 \
SKIP_TRAIN=0 \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/rl/grpo.sh
