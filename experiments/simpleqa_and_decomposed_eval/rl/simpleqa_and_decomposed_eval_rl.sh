#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_simpleqa_and_decomposed_eval_grpo"
EXPERIMENT_NAME="binary_simpleqa_and_decomposed_eval"

MODEL_PATH="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/richqa_base/sft_lr1.5e-4_epmax30_seed1/global_step_1290/merged_hf_model"
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: SFT checkpoint missing merged_hf_model: ${MODEL_PATH}" >&2
    exit 1
fi

echo "[train_rl] Using SFT checkpoint: ${MODEL_PATH}"

DATA_DIR="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/rl/simpleqa_and_decomposed_eval"
TRAIN_DATA="${DATA_DIR}/train.parquet"
EVAL_DATA="${DATA_DIR}/train_eval.parquet ${DATA_DIR}/val.parquet"

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
