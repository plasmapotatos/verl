#!/usr/bin/env bash
# Re-run train_eval_origqa eval on the expanded 1500-row split.
# Sourced by per-dilution wrappers that set DILUTION_TAG (e.g. "0", "0_25").
set -euo pipefail

if [[ -z "${DILUTION_TAG:-}" ]]; then
    echo "ERROR: DILUTION_TAG not set" >&2
    exit 1
fi

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
EXP_DIR="$WORKDIR/outputs/sft/simpleqa_decompose_dilution/simpleqa_decompose_dilution_${DILUTION_TAG}_sft/sft_lr1.5e-4_epmax20_seed1"
EVAL_PARQUET="$WORKDIR/data/simpleqa/partition/decompose_dilution_test/sft/${DILUTION_TAG}/train_eval_origqa.parquet"

if [[ ! -d "$EXP_DIR" ]]; then
    echo "ERROR: experiment dir missing: $EXP_DIR" >&2
    exit 1
fi
if [[ ! -f "$EVAL_PARQUET" ]]; then
    echo "ERROR: eval parquet missing: $EVAL_PARQUET" >&2
    exit 1
fi

echo "[reeval] dilution=$DILUTION_TAG"
echo "[reeval] exp_dir=$EXP_DIR"
echo "[reeval] eval=$EVAL_PARQUET"

cd "$WORKDIR"
N_GPUS_PER_NODE=4 PROMPT_LEN=2048 RESP_LEN=1024 USE_JUDGE=1 \
    bash experiments/utils/eval_extra_data.sh "$EXP_DIR" "$EVAL_PARQUET"
