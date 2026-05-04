#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="correct_plus_novel_bag_grpo"
EXPERIMENT_NAME="correct_plus_novel_bag"

MODEL_PATH="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/simpleqa_factual_anchor_sft_smoketest/sft_lr1.5e-4_epmax30_seed1/global_step_570/merged_hf_model"
TRAIN_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/factual_anchor/smoketest/rl/train.parquet"
EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/factual_anchor/smoketest/rl/train_eval.parquet /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/factual_anchor/smoketest/rl/val.parquet"
PASS_AT_K_EVAL_DATA="$EVAL_DATA"

# Combined reward mode:
# reward = alpha * correct + beta * novel_bag
# alpha and beta default to 1.0 in the scorer.
export NOVEL_BAG_ALPHA=1.0
export NOVEL_BAG_BETA=1.0

PROJECT_NAME="$PROJECT_NAME" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
MODEL_PATH="$MODEL_PATH" \
TRAIN_DATA="$TRAIN_DATA" \
VAL_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
PASS_AT_K_EVAL_DATA="$PASS_AT_K_EVAL_DATA" \
REWARD_MODE=correct_plus_novel_bag \
TOTAL_EPOCHS=40 \
ROLLOUT_N=8 \
SAVE_FREQ=100 \
RUN_EVAL=1 \
SKIP_TRAIN=0 \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/rl/grpo.sh
