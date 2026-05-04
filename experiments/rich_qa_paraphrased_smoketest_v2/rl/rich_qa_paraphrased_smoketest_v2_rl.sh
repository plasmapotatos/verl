#!/usr/bin/env bash
set -euo pipefail

# RL-only smoketest off a pre-existing SFT checkpoint, using the regenerated
# (diversified-prompt) paraphrased smoketest data: 1/4 of the rich_qa ids.

PROJECT_NAME="simpleqa_rich_qa_paraphrased_smoketest_v2_grpo"
EXPERIMENT_NAME="binary_rich_qa_paraphrased_smoketest_v2"

CKPT_DIR="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/richqa_base/sft_lr1.5e-4_epmax30_seed1/global_step_1290"
MODEL_PATH="${CKPT_DIR}/merged_hf_model"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: SFT checkpoint is missing merged_hf_model: ${MODEL_PATH}" >&2
    exit 1
fi

echo "[train_rl] Using SFT checkpoint: ${MODEL_PATH}"

DATA_DIR="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/rl"
TRAIN_DATA="${DATA_DIR}/paraphrased/smoketest/train.parquet"
EVAL_DATA="${DATA_DIR}/paraphrased/smoketest/train_eval.parquet ${DATA_DIR}/val.parquet"

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
