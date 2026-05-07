#!/usr/bin/env bash
set -euo pipefail

# Eval+plot target for the richqa_sdft_natural experiment. Runs under the
# verl apptainer. Generates per-checkpoint predictions on both the rich and
# origqa eval splits, grades with verl's SimpleQA judge, and plots metrics
# vs. step.

OUTPUT_DIR="${OUTPUT_DIR:-/work/hdd/bbsg/twei2/rl/Self-Distillation/outputs/richqa_sdft_natural_3b}"
EVAL_DATA_RICH="${EVAL_DATA_RICH:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/sft/train_eval.parquet}"
EVAL_DATA_ORIG="${EVAL_DATA_ORIG:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/sft/train_eval_origqa.parquet}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

BASE_MODEL="$BASE_MODEL" \
bash /work/hdd/bbsg/twei2/rl/verl/experiments/sdft_eval_and_plot.sh \
    "$OUTPUT_DIR" \
    "$EVAL_DATA_RICH" \
    "$EVAL_DATA_ORIG"

[[ -n "${CHAIN_FLAG_FILE:-}" ]] && touch "$CHAIN_FLAG_FILE"
