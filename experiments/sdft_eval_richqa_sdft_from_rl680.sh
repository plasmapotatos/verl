#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/work/hdd/bbsg/twei2/rl/Self-Distillation/outputs/sdft_richqa_sdft_from_rl680_3b}"
EVAL_DATA="${EVAL_DATA:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/rl/train_eval.parquet /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/rl/val.parquet}"
BASE_MODEL="${BASE_MODEL:-/work/hdd/bbsg/twei2/rl/verl/outputs/rl/richqa_grpo_base/binary/global_step_680/merged_hf_model}"

BASE_MODEL="$BASE_MODEL" \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/sdft_eval_and_plot.sh \
    "$OUTPUT_DIR" \
    $EVAL_DATA

[[ -n "${CHAIN_FLAG_FILE:-}" ]] && touch "$CHAIN_FLAG_FILE"
