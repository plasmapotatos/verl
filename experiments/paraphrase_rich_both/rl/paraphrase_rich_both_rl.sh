#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_paraphrase_rich_both_grpo"
EXPERIMENT_NAME="binary_paraphrase_rich_both"

SFT_PROJECT="simpleqa_paraphrase_rich_both_sft"
SFT_EXP_GLOB="sft_lr1.5e-4_epmax30_seed1"
SFT_ROOT="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/${SFT_PROJECT}/${SFT_EXP_GLOB}"

if [[ -n "${SFT_CKPT_STEP:-}" ]]; then
    CKPT_DIR="${SFT_ROOT}/global_step_${SFT_CKPT_STEP}"
else
    CKPT_DIR="$(ls -d "${SFT_ROOT}"/global_step_* 2>/dev/null | sort -V | tail -n 1)"
fi

if [[ -z "${CKPT_DIR:-}" || ! -d "${CKPT_DIR}" ]]; then
    echo "ERROR: could not find SFT checkpoint under ${SFT_ROOT}" >&2
    echo "Set SFT_CKPT_STEP=<N> to pin a specific step, or run SFT first." >&2
    exit 1
fi

MODEL_PATH="${CKPT_DIR}/merged_hf_model"
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: SFT checkpoint is missing merged_hf_model: ${MODEL_PATH}" >&2
    exit 1
fi

echo "[train_rl] Using SFT checkpoint: ${MODEL_PATH}"

TRAIN_DATA="/work/hdd/bbsg/twei2/rl/verl/shared/data/simpleqa/partition/paraphrase_rich_both/rl/train.parquet"
EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/shared/data/simpleqa/partition/paraphrase_rich_both/rl/train_eval.parquet /work/hdd/bbsg/twei2/rl/verl/shared/data/simpleqa/partition/paraphrase_rich_both/rl/eval.parquet"
PASS_AT_K_EVAL_DATA="$EVAL_DATA"

PROJECT_NAME="$PROJECT_NAME" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
MODEL_PATH="$MODEL_PATH" \
TRAIN_DATA="$TRAIN_DATA" \
VAL_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
PASS_AT_K_EVAL_DATA="$PASS_AT_K_EVAL_DATA" \
REWARD_MODE=binary \
TOTAL_EPOCHS=20 \
ROLLOUT_N=8 \
SAVE_FREQ=100 \
RUN_EVAL=1 \
SKIP_TRAIN=0 \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/rl/grpo.sh
