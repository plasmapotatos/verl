#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_rich_sft_grpo_train_eval"
EXPERIMENT_NAME="simpleqa_rich_sft_grpo_train_eval_clean"

MODEL_PATH="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/simpleqa_rich_sft_clean/sft_lr1.5e-4_ep10_seed1/global_step_280/merged_hf_model"
TRAIN_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/rich_sft_clean/train_origqa_frac0.9.parquet"
EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/rich_sft_clean/train_origqa_frac0.9_frac0.1.parquet /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/rich_sft_clean/train_origqa_frac0.1_eval.parquet"
PASS_AT_K_EVAL_DATA="$EVAL_DATA"

PROJECT_NAME="$PROJECT_NAME" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
MODEL_PATH="$MODEL_PATH" \
TRAIN_DATA="$TRAIN_DATA" \
VAL_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
PASS_AT_K_EVAL_DATA="$PASS_AT_K_EVAL_DATA" \
TOTAL_EPOCHS=20 \
ROLLOUT_N=8 \
SAVE_FREQ=100 \
RUN_EVAL=1 \
SKIP_TRAIN=0 \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/rl/grpo.sh
