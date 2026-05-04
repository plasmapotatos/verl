#!/usr/bin/env bash
set -euo pipefail

bash /work/hdd/bbsg/twei2/rl/verl/experiments/sdft_eval_and_plot.sh \
    /work/hdd/bbsg/twei2/rl/verl/outputs/sdft_simpleqa_window_fixed \
    /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose/sft/train_eval.parquet

# When invoked by chain_job.sh, mark completion so the chain stops resubmitting.
[[ -n "${CHAIN_FLAG_FILE:-}" ]] && touch "$CHAIN_FLAG_FILE"
