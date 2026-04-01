#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_rich_sft_grpo_refusal"
EXPERIMENT_NAME="simpleqa_rich_sft_grpo_refusal_ternary_adaptive"

MODEL_PATH="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/simpleqa_rich_sft_refusal_progressive/sft_lr1.5e-4_epmax30_seed1/global_step_700/merged_hf_model"
TRAIN_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal/rl/train_richqa_rl_combined.parquet"
EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal/rl/train_richqa_answer_origqa_frac0.8_frac0.2.parquet /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal/rl/train_richqa_answer_origqa_frac0.2.parquet /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal/rl/train_richqa_refusal_origqa_frac0.8_frac0.2.parquet /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal/rl/train_richqa_refusal_origqa_frac0.2.parquet"
PASS_AT_K_EVAL_DATA="$EVAL_DATA"

PROJECT_NAME="$PROJECT_NAME" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
MODEL_PATH="$MODEL_PATH" \
TRAIN_DATA="$TRAIN_DATA" \
VAL_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
PASS_AT_K_EVAL_DATA="$PASS_AT_K_EVAL_DATA" \
REWARD_MODE=ternary_adaptive \
TOTAL_EPOCHS=20 \
ROLLOUT_N=8 \
SAVE_FREQ=100 \
RUN_EVAL=1 \
SKIP_TRAIN=0 \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/rl/grpo.sh
