#!/usr/bin/env bash
set -e

VERL_DIR="${VERL_DIR:-/work/hdd/bbsg/twei2/rl/verl}"

EXPERIMENT_DIR="$VERL_DIR/outputs/rl/simpleqa_decompose_grpo/binary_decompose"
EVAL_DATA=(
	"$VERL_DIR/data/simpleqa/partition/decompose/rl/train_eval_decompose_combined.parquet"
	"$VERL_DIR/data/simpleqa/partition/decompose/rl/val_decompose_combined.parquet"
)

bash "$VERL_DIR/experiments/utils/eval_extra_data.sh" \
	"$EXPERIMENT_DIR" \
	"${EVAL_DATA[@]}"
