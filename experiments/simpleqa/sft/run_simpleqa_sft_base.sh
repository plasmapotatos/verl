#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_sft_base_sweep"

TRAIN_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/base/train.parquet"
EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/base/train_frac0.1.parquet"

PROJECT_NAME="$PROJECT_NAME" \
TRAIN_DATA="$TRAIN_DATA" \
PROMPT_KEY="question" \
RESPONSE_KEY="answer" \
PROMPT_DICT_KEYS="" \
RESPONSE_DICT_KEYS="" \
bash experiments/sft/run_sft_sweep_with_eval.sh