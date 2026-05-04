#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_factual_anchor_grpo_smoketest"
EXPERIMENT_NAME="binary"

# TODO: update MODEL_PATH to best checkpoint after SFT training completes
MODEL_PATH="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/simpleqa_factual_anchor_sft_smoketest/sft_lr1.5e-4_epmax30_seed1/global_step_570/merged_hf_model"

TRAIN_DATA="data/simpleqa/partition/factual_anchor/smoketest/rl/train.parquet"

# fit: samples seen during training; generalization: held-out eval set
EVAL_DATA="data/simpleqa/partition/factual_anchor/smoketest/rl/train_eval.parquet data/simpleqa/partition/factual_anchor/smoketest/rl/val.parquet"
PASS_AT_K_EVAL_DATA="$EVAL_DATA"

PROJECT_NAME="$PROJECT_NAME" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
MODEL_PATH="$MODEL_PATH" \
TRAIN_DATA="$TRAIN_DATA" \
VAL_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
PASS_AT_K_EVAL_DATA="$PASS_AT_K_EVAL_DATA" \
REWARD_MODE=binary \
TOTAL_EPOCHS=40 \
ROLLOUT_N=8 \
SAVE_FREQ=100 \
RUN_EVAL=1 \
SKIP_TRAIN=0 \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/rl/grpo.sh
