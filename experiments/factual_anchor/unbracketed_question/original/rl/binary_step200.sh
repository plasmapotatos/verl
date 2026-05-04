#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="factual_anchor_grpo_unbracketed_question_original"
EXPERIMENT_NAME="binary_unbracketed_question_original_step200"

MODEL_PATH="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/simpleqa_factual_anchor_sft_unbracketed_question_original/sft_lr1.5e-4_epmax30_seed1/global_step_200/merged_hf_model"

TRAIN_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/factual_anchor/smoketest/unbracketed/original/rl/train.parquet"
EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/factual_anchor/smoketest/unbracketed/original/rl/train_eval.parquet /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/factual_anchor/smoketest/unbracketed/original/rl/val.parquet"
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
