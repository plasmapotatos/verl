#!/usr/bin/env bash
# Run global_step_4140 on the CoT variant of train_eval_k3.
# Bumps response length since chain-of-thought generations are longer.

set -euo pipefail

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
EXPERIMENT_DIR="$WORKDIR/outputs/sft/simpleqa_decompose_mixed_k12_smoke_sft/sft_lr1.5e-4_epmax30_seed1"
CKPT_DIR="$EXPERIMENT_DIR/global_step_4140"
MERGED="$CKPT_DIR/merged_hf_model"
EXPERIMENT_NAME="$(basename "$EXPERIMENT_DIR")"
STEP_NAME="global_step_4140"

DATASET="${DATASET:-simpleqa}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
RESP_LEN="${RESP_LEN:-1024}"
USE_JUDGE="${USE_JUDGE:-1}"

EVAL_PATH="$WORKDIR/data/simpleqa/partition/decompose_mixed_k12_smoke/sft/train_eval_k3_cot.parquet"

if [[ ! -f "$EVAL_PATH" ]]; then
    echo "ERROR: eval data not found: $EVAL_PATH" >&2
    exit 1
fi
if [[ ! -f "$MERGED/config.json" ]]; then
    bash "$WORKDIR/experiments/utils/merge_checkpoint.sh" "$CKPT_DIR"
fi

GEN_DIR="$CKPT_DIR/generations"
mkdir -p "$GEN_DIR"
EVAL_TAG="$(basename "${EVAL_PATH%.parquet}")"
GEN_OUT="$GEN_DIR/${EXPERIMENT_NAME}_${STEP_NAME}__on_${EVAL_TAG}.parquet"
EVAL_OUT="${GEN_OUT%.parquet}_eval.json"

echo "[$STEP_NAME] eval_data=$(basename "$EVAL_PATH")"
if [[ -f "$GEN_OUT" && -s "$GEN_OUT" ]]; then
    echo "  generation exists: $(basename "$GEN_OUT")"
else
    echo "  generating -> $GEN_OUT"
    python -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
        data.path="$EVAL_PATH" \
        data.prompt_key=prompt \
        data.n_samples=1 \
        data.output_path="$GEN_OUT" \
        model.path="$MERGED" \
        +model.trust_remote_code=True \
        rollout.temperature=0 \
        rollout.prompt_length="$PROMPT_LEN" \
        rollout.response_length="$RESP_LEN" \
        rollout.tensor_model_parallel_size=1 \
        rollout.gpu_memory_utilization=0.8
fi

if [[ -f "$EVAL_OUT" ]]; then
    echo "  eval exists: $(basename "$EVAL_OUT")"
elif [[ ! -f "$GEN_OUT" || ! -s "$GEN_OUT" ]]; then
    echo "  generation missing/empty, skipping eval"
else
    if [[ "$USE_JUDGE" == "1" ]]; then
        python -m verl.eval.cli --dataset "$DATASET" --input "$GEN_OUT" --output "$EVAL_OUT" --use-judge
    else
        python -m verl.eval.cli --dataset "$DATASET" --input "$GEN_OUT" --output "$EVAL_OUT"
    fi
fi

echo "Done."
