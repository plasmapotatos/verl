#!/usr/bin/env bash
set -e

# Run eval_extra_data.sh on a single decompose_dilution experiment using the
# origqa longform eval set. Pass the dilution suffix as $1 (e.g. 0, 0_25, 1).

DILUTION="${1:?usage: $0 <dilution_suffix>  e.g. 0 | 0_25 | 0_5 | 0_75 | 1}"

VERL_DIR="${VERL_DIR:-/work/hdd/bbsg/twei2/rl/verl}"
ROOT="$VERL_DIR/outputs/rl/simpleqa_decompose_dilution"
EVAL_DATA="$VERL_DIR/data/simpleqa/partition/decompose_dilution_test/sft/0/train_eval_origqa_550_longform.parquet"

EXP_DIR="$ROOT/simpleqa_decompose_dilution_${DILUTION}_grpo/binary_decompose_dilution_${DILUTION}"

if [[ ! -f "$EVAL_DATA" ]]; then
	echo "ERROR: eval data not found: $EVAL_DATA" >&2
	exit 1
fi
if [[ ! -d "$EXP_DIR" ]]; then
	echo "ERROR: experiment dir not found: $EXP_DIR" >&2
	exit 1
fi

echo "=== dilution=${DILUTION} ==="
bash "$VERL_DIR/experiments/utils/eval_extra_data.sh" "$EXP_DIR" "$EVAL_DATA"
