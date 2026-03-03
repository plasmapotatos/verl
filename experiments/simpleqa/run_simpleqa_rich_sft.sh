
#!/usr/bin/env bash
set -euo pipefail

MODE="simpleqa_rich_sft"

PROJECT_NAME="simpleqa_sft_aug__${MODE}"

TRAIN_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/${MODE}_train.parquet"

EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/base/train.parquet"

PROJECT_NAME="$PROJECT_NAME" \
TRAIN_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
PROMPT_KEY="question" \
RESPONSE_KEY="answer" \
PROMPT_DICT_KEYS="" \
RESPONSE_DICT_KEYS="" \
MAX_LENGTH="4096" \
FILTER_OVERLONG_PROMPTS="1" \
LR_LIST="1e-4" \
EPOCHS_LIST="1 3 6 10" \
bash experiments/sft/run_sft_sweep_with_eval.sh