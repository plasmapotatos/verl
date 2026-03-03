#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_sft_aug_direct"

TRAIN_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/llm_paraphrase_train_direct.parquet"

EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/direct/train.parquet"

PROJECT_NAME="$PROJECT_NAME" \
TRAIN_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
bash experiments/sft/run_sft_sweep_with_eval.sh