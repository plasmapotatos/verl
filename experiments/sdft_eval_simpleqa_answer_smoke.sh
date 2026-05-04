#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/work/hdd/bbsg/twei2/rl/verl/outputs/sdft_simpleqa_answer_smoke_3b}"
EVAL_DATA="${EVAL_DATA:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose/sft/train_eval.parquet}"

bash /work/hdd/bbsg/twei2/rl/verl/experiments/sdft_eval_and_plot.sh \
    "$OUTPUT_DIR" \
    "$EVAL_DATA"

# When invoked by chain_job.sh, mark completion so the chain stops resubmitting.
[[ -n "${CHAIN_FLAG_FILE:-}" ]] && touch "$CHAIN_FLAG_FILE"
