#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="richqa_spin_dpo_base"
EXPERIMENT_NAME="spin_dpo"

MODEL_PATH="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/richqa_base/sft_lr1.5e-4_epmax30_seed1/global_step_1290/merged_hf_model"
TRAIN_DATA="data/simpleqa/partition/rich_qa/rl/train.parquet"
EVAL_DATA="data/simpleqa/partition/rich_qa/rl/train_eval.parquet data/simpleqa/partition/rich_qa/rl/val.parquet"

PROJECT_NAME="$PROJECT_NAME" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
MODEL_PATH="$MODEL_PATH" \
TRAIN_DATA="$TRAIN_DATA" \
VAL_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
TOTAL_EPOCHS=20 \
ROLLOUT_N=2 \
SAVE_FREQ=100 \
RUN_EVAL=1 \
SKIP_TRAIN=0 \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/rl/spin_dpo.sh
