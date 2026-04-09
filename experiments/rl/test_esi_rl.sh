#!/usr/bin/env bash
# Smoke test: verify ESI forced checkpoint save works during actual RL training.
#
# Uses the factual_anchor_smoketest_original config (known good model + data).
# Sets SAVE_FREQ=9999 so normal periodic saves NEVER trigger.
# If any global_step_* checkpoint appears, it MUST be from ESI.
#
# Usage:
#   ./chain_job.sh --gpus 2 --time 00:10:00 experiments/rl/test_esi_rl.sh
#
# After it runs, check:
#   grep -i "force saving" logs/chain/test_esi_rl/iter_*.out
#   ls outputs/rl/esi_test/esi_smoke/global_step_*
#
# Clean up after:
#   rm -rf outputs/rl/esi_test
set -euo pipefail

PROJECT_NAME="esi_test"
EXPERIMENT_NAME="esi_smoke"

MODEL_PATH="outputs/sft/simpleqa_factual_anchor_sft_smoketest_original/sft_lr1.5e-4_epmax30_seed1/global_step_570/merged_hf_model"
TRAIN_DATA="data/simpleqa/partition/factual_anchor/smoketest/original/rl/train.parquet"
EVAL_DATA="data/simpleqa/partition/factual_anchor/smoketest/original/rl/train_eval.parquet"

PROJECT_NAME="$PROJECT_NAME" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
MODEL_PATH="$MODEL_PATH" \
TRAIN_DATA="$TRAIN_DATA" \
VAL_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
REWARD_MODE=binary \
TOTAL_EPOCHS=20 \
ROLLOUT_N=4 \
SAVE_FREQ=50 \
RUN_EVAL=0 \
RUN_BASE_EVAL=0 \
RUN_BASE_PASS_AT_K=0 \
PRUNE_CHECKPOINTS=0 \
SKIP_TRAIN=0 \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/rl/grpo.sh
