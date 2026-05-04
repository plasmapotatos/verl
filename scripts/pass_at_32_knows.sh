#!/usr/bin/env bash
# Pass@32 on knows.parquet for the post-SFT and post-RL checkpoints
# (i.e. does the model still get right what the base Qwen knew?).
set -euo pipefail
cd /work/hdd/bbsg/twei2/rl/verl

EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/knows.parquet"
TOP_K="${TOP_K:-32}"
TOP_P="${TOP_P:-0.9}"
TEMPERATURE="${TEMPERATURE:-1}"
N_GPUS="${N_GPUS_PER_NODE:-1}"

declare -A CKPTS=(
    [base_qwen]="Qwen/Qwen2.5-3B-Instruct"
    [sft_richqa_refusal_50_50]="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/simpleqa_rich_sft_refusal_partition_50_50/sft_lr1.5e-4_epmax30_seed1/global_step_1290/merged_hf_model"
    [rl_richqa_grpo_refusal_50_50]="/work/hdd/bbsg/twei2/rl/verl/outputs/rl/richqa_grpo_refusal_50_50/binary/global_step_680/merged_hf_model"
)

OUT_ROOT="/work/hdd/bbsg/twei2/rl/verl/outputs/analysis/pass_at_${TOP_K}_knows"
mkdir -p "$OUT_ROOT"

for label in "${!CKPTS[@]}"; do
    ckpt="${CKPTS[$label]}"
    out_dir="$OUT_ROOT/$label"
    mkdir -p "$out_dir"
    if [[ -f "$out_dir/pass@k/pass_at_k_${TOP_K}.json" ]]; then
        echo "=== pass@${TOP_K} :: $label already done, skipping ==="
        continue
    fi
    echo "=== pass@${TOP_K} :: $label -> $ckpt ==="
    python scripts/pass_at_k.py \
        --checkpoint "$ckpt" \
        --dataset simpleqa \
        --eval-data "$EVAL_DATA" \
        --output-dir "$out_dir" \
        --top-k "$TOP_K" \
        --top-p "$TOP_P" \
        --temperature "$TEMPERATURE" \
        --n-gpus "$N_GPUS" \
        --tp-size "$N_GPUS" \
        --use-judge
done
