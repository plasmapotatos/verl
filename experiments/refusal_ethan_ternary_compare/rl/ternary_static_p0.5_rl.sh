#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_refusal_ethan_ternary_compare_grpo"
EXPERIMENT_NAME="ternary_static_p0.5_refusal_ethan_ternary_compare"

SFT_PROJECT="simpleqa_refusal_ethan_ternary_compare_sft"
SFT_EXP_GLOB="sft_lr1.5e-4_epmax20_seed1"
SFT_ROOT="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/${SFT_PROJECT}/${SFT_EXP_GLOB}"

if [[ -n "${SFT_CKPT_STEP:-}" ]]; then
    CKPT_DIR="${SFT_ROOT}/global_step_${SFT_CKPT_STEP}"
else
    CKPT_DIR="$(ls -d "${SFT_ROOT}"/global_step_* 2>/dev/null | sort -V | tail -n 1)"
fi

if [[ -z "${CKPT_DIR:-}" || ! -d "${CKPT_DIR}" ]]; then
    echo "ERROR: could not find SFT checkpoint under ${SFT_ROOT}" >&2
    exit 1
fi

MODEL_PATH="${CKPT_DIR}/merged_hf_model"
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: SFT checkpoint is missing merged_hf_model: ${MODEL_PATH}" >&2
    exit 1
fi

echo "[ternary_static_rl] Using SFT checkpoint: ${MODEL_PATH}"

DATA_DIR="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/refusal_ethan/rl"
TRAIN_DATA="$DATA_DIR/train.parquet"
EVAL_DATA="$DATA_DIR/answer_eval.parquet $DATA_DIR/refuse_eval.parquet $DATA_DIR/train_answer_eval.parquet $DATA_DIR/train_refuse_eval.parquet"

PROJECT_NAME="$PROJECT_NAME" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
MODEL_PATH="$MODEL_PATH" \
TRAIN_DATA="$TRAIN_DATA" \
VAL_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
REWARD_MODE=ternary_static \
TERNARY_PENALTY=0.5 \
TOTAL_EPOCHS=10 \
ROLLOUT_N=8 \
SAVE_FREQ=50 \
RUN_EVAL=1 \
SKIP_TRAIN=0 \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/rl/grpo.sh
