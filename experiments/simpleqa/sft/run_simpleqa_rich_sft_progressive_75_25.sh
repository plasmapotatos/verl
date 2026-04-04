#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="simpleqa_rich_sft_refusal_progressive_75_25"

TRAIN_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal/sft/75-25/train_richqa_combined.parquet"

EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal/sft/75-25/train_richqa_answer_frac0.2.parquet /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal/sft/75-25/train_richqa_answer_origqa_frac0.2.parquet /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal/sft/75-25/train_richqa_refusal_frac0.2.parquet /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal/sft/75-25/train_richqa_refusal_origqa_frac0.2.parquet"

PASS_AT_K_EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal/sft/75-25/train_richqa_answer_origqa_frac0.2.parquet /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal/sft/75-25/train_richqa_refusal_origqa_frac0.2.parquet"

PROJECT_NAME="$PROJECT_NAME" \
TRAIN_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
PASS_AT_K_EVAL_DATA="$PASS_AT_K_EVAL_DATA" \
PROMPT_KEY="question" \
RESPONSE_KEY="answer" \
PROMPT_DICT_KEYS="" \
RESPONSE_DICT_KEYS="" \
MAX_LENGTH="4096" \
FILTER_OVERLONG_PROMPTS="1" \
LR_LIST="1.5e-4" \
EPOCHS_LIST="1 3 6 10 20 30" \
bash experiments/sft/run_sft_progressive_eval.sh
