#!/usr/bin/env bash
# Pass@32 for simpleqa_decompose_sft global_step_7860 on
# train_eval_decompose_combined.parquet, via scripts/pass_at_k.py.

set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set (judge step will fail)" >&2
    exit 1
fi

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
CKPT="$WORKDIR/outputs/sft/simpleqa_decompose_sft/sft_lr1.5e-4_epmax30_seed1/global_step_7860/merged_hf_model"
EVAL_DATA="$WORKDIR/data/simpleqa/partition/decompose/rl/train_eval_decompose_combined.parquet"
OUT_DIR="$WORKDIR/outputs/sft/simpleqa_decompose_sft/sft_lr1.5e-4_epmax30_seed1/global_step_7860/pass_at_k_train_eval_decompose_combined"

K="${K:-32}"
N_GPUS="${N_GPUS:-4}"
TP_SIZE="${TP_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
RESP_LEN="${RESP_LEN:-256}"

mkdir -p "$OUT_DIR"

python "$WORKDIR/scripts/pass_at_k.py" \
    --checkpoint "$CKPT" \
    --dataset simpleqa \
    --eval-data "$EVAL_DATA" \
    --output-dir "$OUT_DIR" \
    --top-k "$K" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --prompt-len "$PROMPT_LEN" \
    --resp-len "$RESP_LEN" \
    --n-gpus "$N_GPUS" \
    --tp-size "$TP_SIZE" \
    --use-judge

echo "Summary: $OUT_DIR/pass@k/pass_at_k_${K}.json"
