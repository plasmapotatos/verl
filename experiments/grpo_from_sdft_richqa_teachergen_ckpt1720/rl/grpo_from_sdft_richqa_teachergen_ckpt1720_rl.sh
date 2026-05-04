#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_grpo_from_sdft_richqa_teachergen_ckpt1720_grpo"
EXPERIMENT_NAME="binary_grpo_from_sdft_richqa_teachergen_ckpt1720"

MODEL_PATH="/work/hdd/bbsg/twei2/rl/Self-Distillation/outputs/sdft_richqa_full_3b_teachergen/checkpoint-1720"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: SDFT checkpoint not found: ${MODEL_PATH}" >&2
    exit 1
fi

echo "[train_rl] Using SDFT checkpoint: ${MODEL_PATH}"

TRAIN_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/rl/train.parquet"
EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/rl/train_eval.parquet /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/rl/val.parquet"

PROJECT_NAME="$PROJECT_NAME" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
MODEL_PATH="$MODEL_PATH" \
TRAIN_DATA="$TRAIN_DATA" \
VAL_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
REWARD_MODE=binary \
TOTAL_EPOCHS=20 \
ROLLOUT_N=8 \
SAVE_FREQ=100 \
RUN_EVAL=1 \
SKIP_TRAIN=0 \
TIME_BUDGET_SEC=7200 \
TIME_BUDGET_BUFFER_SEC=600 \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/rl/grpo.sh
