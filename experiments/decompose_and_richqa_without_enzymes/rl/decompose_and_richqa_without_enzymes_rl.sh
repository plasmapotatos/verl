#!/usr/bin/env bash
# GRPO on top of the decompose_and_richqa_without_enzymes-SFT checkpoint, using the combined RL split.
set -euo pipefail

PROJECT_NAME="simpleqa_decompose_and_richqa_without_enzymes_grpo"
EXPERIMENT_NAME="binary_decompose_and_richqa_without_enzymes"

# Resolve the most recent SFT checkpoint produced by the matching SFT script.
# The SFT project name and experiment name must match run_sft_progressive_eval.sh's layout:
#   outputs/sft/{PROJECT_NAME}/sft_lr{LR}_epmax{MAX_EPOCHS}_seed{SEED}/global_step_*
SFT_PROJECT="simpleqa_decompose_and_richqa_without_enzymes_sft"
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

TRAIN_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose_and_richqa_without_enzymes/rl/train.parquet"
EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose_and_richqa_without_enzymes/rl/train_eval.parquet /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose_and_richqa_without_enzymes/rl/val.parquet"
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
