#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="richqa_grpo"
EXPERIMENT_NAME="sdft_on_richqa_binary_reward"

MODEL_PATH="/work/hdd/bbsg/twei2/rl/Self-Distillation/outputs/sdft_richqa/checkpoint-154"
TRAIN_DATA="data/simpleqa/partition/rich_qa/rl/train.parquet"
EVAL_DATA="data/simpleqa/partition/rich_qa/rl/train_eval.parquet data/simpleqa/partition/rich_qa/rl/val.parquet"
PASS_AT_K_EVAL_DATA="$EVAL_DATA"

PROJECT_NAME="$PROJECT_NAME" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
MODEL_PATH="$MODEL_PATH" \
TRAIN_DATA="$TRAIN_DATA" \
VAL_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
PASS_AT_K_EVAL_DATA="$PASS_AT_K_EVAL_DATA" \
REWARD_MODE=binary \
TOTAL_EPOCHS=20 \
ROLLOUT_N=8 \
SAVE_FREQ=100 \
RUN_EVAL=1 \
SKIP_TRAIN=0 \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/rl/grpo.sh
