#!/usr/bin/env bash
set -euo pipefail

MODE="completion"

PROJECT_NAME="simpleqa_sft_aug_${MODE}_base_sweep"

TRAIN_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/${MODE}_train.parquet"

EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/base/train.parquet"
PROJECT_NAME="$PROJECT_NAME" \
TRAIN_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
bash experiments/sft/run_sft_sweep_with_eval.sh