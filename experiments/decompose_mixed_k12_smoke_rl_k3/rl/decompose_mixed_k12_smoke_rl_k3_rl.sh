#!/usr/bin/env bash
# RL-only run: GRPO on top of the decompose_mixed_k12_smoke SFT checkpoint
# (step 4140), training on k=3 combined Q/A and evaluating on both k=3 and
# k=4 train_eval / val splits in rl_k3/.
set -euo pipefail

PROJECT_NAME="simpleqa_decompose_mixed_k12_smoke_rl_k3_grpo"
EXPERIMENT_NAME="binary_decompose_mixed_k12_smoke_rl_k3"

# Reuse the existing k12 SFT checkpoint at step 4140.
SFT_PROJECT="simpleqa_decompose_mixed_k12_smoke_sft"
SFT_EXP_GLOB="sft_lr1.5e-4_epmax30_seed1"
SFT_ROOT="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/${SFT_PROJECT}/${SFT_EXP_GLOB}"
SFT_CKPT_STEP="${SFT_CKPT_STEP:-4140}"
CKPT_DIR="${SFT_ROOT}/global_step_${SFT_CKPT_STEP}"

if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "ERROR: SFT checkpoint not found at ${CKPT_DIR}" >&2
    exit 1
fi

MODEL_PATH="${CKPT_DIR}/merged_hf_model"
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: SFT checkpoint is missing merged_hf_model: ${MODEL_PATH}" >&2
    exit 1
fi

echo "[train_rl] Using SFT checkpoint: ${MODEL_PATH}"

DATA_ROOT="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose_mixed_k12_smoke/rl_k3"

TRAIN_DATA="$DATA_ROOT/train.parquet"
EVAL_DATA="$DATA_ROOT/train_eval.parquet $DATA_ROOT/train_eval_k4.parquet $DATA_ROOT/val.parquet $DATA_ROOT/val_k4.parquet"
PASS_AT_K_EVAL_DATA="$EVAL_DATA"

PROJECT_NAME="$PROJECT_NAME" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
MODEL_PATH="$MODEL_PATH" \
TRAIN_DATA="$TRAIN_DATA" \
VAL_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
PASS_AT_K_EVAL_DATA="" \
REWARD_MODE=binary \
TOTAL_EPOCHS=20 \
ROLLOUT_N=8 \
SAVE_FREQ=100 \
RUN_EVAL=1 \
SKIP_TRAIN=0 \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/rl/grpo.sh
