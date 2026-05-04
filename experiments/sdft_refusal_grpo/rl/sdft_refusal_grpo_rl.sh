#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_sdft_refusal_grpo_grpo"
EXPERIMENT_NAME="binary_sdft_refusal_grpo"

MODEL_PATH="/work/hdd/bbsg/twei2/rl/Self-Distillation/outputs/sdft_refusal_on_sft_3b/checkpoint-2720"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: model path does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

echo "[train_rl] Using model: ${MODEL_PATH}"

DATA_DIR="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/refusal/rl"
TRAIN_DATA="${DATA_DIR}/train.parquet"
EVAL_DATA="${DATA_DIR}/train_eval_answer.parquet ${DATA_DIR}/train_eval_refusal.parquet ${DATA_DIR}/val_answer.parquet ${DATA_DIR}/val_refusal.parquet"

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
bash /work/hdd/bbsg/twei2/rl/verl/experiments/rl/grpo.sh
