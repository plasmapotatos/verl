#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_sft_base_sweep"

TRAIN_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/base/train.parquet"

PROJECT_NAME="$PROJECT_NAME" \
TRAIN_DATA="$TRAIN_DATA" \
bash experiments/sft/run_sft_sweep_with_eval.sh