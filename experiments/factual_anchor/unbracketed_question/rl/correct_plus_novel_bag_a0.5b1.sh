#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="factual_anchor_grpo_unbracketed_question"
EXPERIMENT_NAME="correct_plus_novel_bag_a0.5b1_unbracketed_question"

MODEL_PATH="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/simpleqa_factual_anchor_sft_smoketest_unbracketed_question/sft_lr1.5e-4_epmax30_seed1/global_step_570/merged_hf_model"

TRAIN_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/factual_anchor/smoketest/unbracketed/rl/train.parquet"
EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/factual_anchor/smoketest/unbracketed/rl/train_eval.parquet /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/factual_anchor/smoketest/unbracketed/rl/val.parquet"
PASS_AT_K_EVAL_DATA="$EVAL_DATA"

export NOVEL_BAG_ALPHA=0.5
export NOVEL_BAG_BETA=1.0
export FACTUAL_ANCHOR_TXT_PATH="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/factual_anchor/smoketest/unbracketed/train_factual_anchors_filtered.txt"

PROJECT_NAME="$PROJECT_NAME" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
MODEL_PATH="$MODEL_PATH" \
TRAIN_DATA="$TRAIN_DATA" \
VAL_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
PASS_AT_K_EVAL_DATA="$PASS_AT_K_EVAL_DATA" \
REWARD_MODE=correct_plus_novel_bag \
TOTAL_EPOCHS=20 \
ROLLOUT_N=8 \
SAVE_FREQ=100 \
RUN_EVAL=1 \
SKIP_TRAIN=0 \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/rl/grpo.sh
