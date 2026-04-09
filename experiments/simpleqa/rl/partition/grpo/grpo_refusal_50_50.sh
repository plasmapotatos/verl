#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="richqa_grpo_refusal_50_50"
EXPERIMENT_NAME="ternary_static"

SFT_DIR="${SFT_DIR:-/work/hdd/bbsg/twei2/rl/verl/outputs/sft/simpleqa_rich_sft_refusal_partition_50_50/sft_lr1.5e-4_epmax30_seed1}"
TRAIN_DATA="data/simpleqa/partition/refusal/rl/train.parquet"
EVAL_DATA="data/simpleqa/partition/refusal/rl/train_eval_answer.parquet data/simpleqa/partition/refusal/rl/train_eval_refusal.parquet data/simpleqa/partition/refusal/rl/val_answer.parquet data/simpleqa/partition/refusal/rl/val_refusal.parquet"
PASS_AT_K_EVAL_DATA="$EVAL_DATA"

PROJECT_NAME="$PROJECT_NAME" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
SFT_DIR="$SFT_DIR" \
TRAIN_DATA="$TRAIN_DATA" \
VAL_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
PASS_AT_K_EVAL_DATA="$PASS_AT_K_EVAL_DATA" \
REWARD_MODE=ternary_static \
TOTAL_EPOCHS=20 \
ROLLOUT_N=8 \
SAVE_FREQ=100 \
RUN_EVAL=1 \
SKIP_TRAIN=0 \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/rl/grpo.sh
