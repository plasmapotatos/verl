#!/usr/bin/env bash
# Run eval on all parquets in the generations folder using gpt-4o-mini judge.
# Outputs: *_eval_mini.json alongside each parquet.
# Usage: bash run_eval_mini.sh

set -euo pipefail

GENDIR="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/simpleqa_rich_sft_refusal_progressive/sft_lr1.5e-4_epmax30_seed1/global_step_930/generations"
CONTAINER="/work/hdd/bbsg/twei2/rl/torch2501.sif"
LOG="/tmp/eval_mini_all.log"

echo "Logging to $LOG"

for parquet in "$GENDIR"/*.parquet; do
    base=$(basename "$parquet" .parquet)
    output="$GENDIR/${base}_eval_mini.json"
    echo "=== $(date) Starting: $base ===" | tee -a "$LOG"
    python -m verl.eval.cli \
        --dataset simpleqa \
        --input "$parquet" \
        --output "$output" \
        --use-judge \
        --judge-model gpt-4o-mini \
        --workers 32 \
        2>&1 | tee -a "$LOG"
    echo "=== Done: $output ===" | tee -a "$LOG"
done

echo "ALL DONE at $(date)" | tee -a "$LOG"
