#!/usr/bin/env bash
# Evaluate multiple checkpoints on an eval parquet, then grade with LLM judge.
# Edit EVAL_DATA and CHECKPOINTS below, then run:
#   bash eval_step480_comparison.sh
# Override hardware via env: NGPU=2 bash eval_step480_comparison.sh
set -euo pipefail

VERL_DIR="${VERL_DIR:-/work/hdd/bbsg/twei2/rl/verl}"
NGPU="${NGPU:-1}"
PROMPT_LEN="${PROMPT_LEN:-2048}"
RESP_LEN="${RESP_LEN:-1024}"

# ---------- CONFIGURE BELOW ----------

EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/refusal/eval.parquet"
OUT_DIR="$VERL_DIR/outputs/eval/step480_comparison"

# Format: "short_name|/path/to/model"  (one per line)
CHECKPOINTS=(
    "base_qwen|Qwen/Qwen2.5-VL-3B-Instruct"
    "base_sft|/work/hdd/bbsg/twei2/rl/verl/outputs/sft/simpleqa_rich_sft_refusal_progressive/sft_lr1.5e-4_epmax30_seed1/global_step_930/merged_hf_model"
    "grpo_binary|/work/hdd/bbsg/twei2/rl/verl/outputs/rl/simpleqa_rich_sft_grpo_refusal/simpleqa_rich_sft_grpo_refusal/global_step_480/merged_hf_model"
    "grpo_ternary_static|/work/hdd/bbsg/twei2/rl/verl/outputs/rl/simpleqa_rich_sft_grpo_refusal/simpleqa_rich_sft_grpo_refusal_ternary_static/global_step_480/merged_hf_model"
    "grpo_ternary_adaptive|/work/hdd/bbsg/twei2/rl/verl/outputs/rl/simpleqa_rich_sft_grpo_refusal/simpleqa_rich_sft_grpo_refusal_ternary_adaptive/global_step_480/merged_hf_model"
)

# ---------- END CONFIG ----------

mkdir -p "$OUT_DIR"

for ENTRY in "${CHECKPOINTS[@]}"; do
    NAME="${ENTRY%%|*}"
    MODEL="${ENTRY##*|}"
    GEN_OUT="$OUT_DIR/${NAME}_gen.parquet"
    EVAL_OUT="$OUT_DIR/${NAME}_eval.json"

    echo ""
    echo "=== $NAME ==="
    echo "    model : $MODEL"
    echo "    eval  : $EVAL_DATA"

    if [[ ! -f "$GEN_OUT" || ! -s "$GEN_OUT" ]]; then
        echo "Generating -> $GEN_OUT"
        python3 -m verl.trainer.main_generation \
            trainer.nnodes=1 \
            trainer.n_gpus_per_node="$NGPU" \
            data.path="$EVAL_DATA" \
            data.prompt_key=prompt \
            data.n_samples=1 \
            data.output_path="$GEN_OUT" \
            model.path="$MODEL" \
            +model.trust_remote_code=True \
            rollout.temperature=0 \
            rollout.prompt_length="$PROMPT_LEN" \
            rollout.response_length="$RESP_LEN" \
            rollout.tensor_model_parallel_size=1 \
            rollout.gpu_memory_utilization=0.8
    else
        echo "Generation already exists, skipping: $GEN_OUT"
    fi

    if [[ ! -f "$EVAL_OUT" ]]; then
        echo "Grading -> $EVAL_OUT"
        python3 -m verl.eval.cli \
            --dataset simpleqa \
            --input "$GEN_OUT" \
            --output "$EVAL_OUT" \
            --use-judge
    else
        echo "Eval already exists, skipping: $EVAL_OUT"
    fi

    echo "Done: $EVAL_OUT"
done

echo ""
echo "All results in: $OUT_DIR"
echo "To plot confusion matrices:"
echo "  python3 $VERL_DIR/scripts/eval_confusion_matrix.py $OUT_DIR/*_eval.json --output $OUT_DIR/confusion_all.png"
