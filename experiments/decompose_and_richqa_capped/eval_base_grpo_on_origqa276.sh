#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate richqa_grpo_base/binary checkpoints on the capped val set
# (276-item original SimpleQA val) so they are head-to-head comparable with
# the binary_decompose_and_richqa_capped_k{1,3,5} runs.
#
# The capped val is symlinked to a unique filename to avoid colliding with
# existing on_val.parquet outputs.

CKPT_ROOT="/work/hdd/bbsg/twei2/rl/verl/outputs/rl/richqa_grpo_base/binary"
EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/_eval_aliases/val_origqa276.parquet"

# Required by eval_all_checkpoints.sh for global_step_0 (uses SFT base).
export MODEL_PATH="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/richqa_base/sft_lr1.5e-4_epmax30_seed1/global_step_1290/merged_hf_model"
export EXPERIMENT_NAME="binary"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"

bash /work/hdd/bbsg/twei2/rl/verl/experiments/utils/eval_all_checkpoints.sh \
    "$CKPT_ROOT" \
    "$EVAL_DATA" \
    "" \
    "$N_GPUS_PER_NODE" \
    2048 \
    512 \
    1
