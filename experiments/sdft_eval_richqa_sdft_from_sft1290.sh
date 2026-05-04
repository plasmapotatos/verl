#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/work/hdd/bbsg/twei2/rl/Self-Distillation/outputs/sdft_richqa_sdft_from_sft1290_3b}"
EVAL_DATA_TRAIN_EVAL="${EVAL_DATA_TRAIN_EVAL:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/rl/train_eval.parquet}"
EVAL_DATA_VAL="${EVAL_DATA_VAL:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/rl/val.parquet}"
BASE_MODEL="${BASE_MODEL:-/work/hdd/bbsg/twei2/rl/verl/outputs/sft/richqa_base/sft_lr1.5e-4_epmax30_seed1/global_step_1290/merged_hf_model}"

BASE_MODEL="$BASE_MODEL" \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/sdft_eval_and_plot.sh \
    "$OUTPUT_DIR" \
    "$EVAL_DATA_TRAIN_EVAL" \
    "$EVAL_DATA_VAL"

[[ -n "${CHAIN_FLAG_FILE:-}" ]] && touch "$CHAIN_FLAG_FILE"
