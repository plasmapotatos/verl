#!/usr/bin/env bash
set -e

MERGED_DIR="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/richqa/sft_lr1.5e-4_epmax30_seed1/global_step_1290/merged_hf_model"
EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/sft/train_origqa_plus_richqa_prefixes_separate.parquet"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
PROMPT_LEN="${PROMPT_LEN:-2048}"
RESP_LEN="${RESP_LEN:-512}"

CKPT_DIR="$(dirname "$MERGED_DIR")"
GEN_DIR="$CKPT_DIR/generations"
mkdir -p "$GEN_DIR"

EVAL_TAG="$(basename "${EVAL_DATA%.parquet}")"
STEP_NAME="$(basename "$CKPT_DIR")"
GEN_OUT="$GEN_DIR/${STEP_NAME}__on_${EVAL_TAG}.parquet"
EVAL_OUT="${GEN_OUT%.parquet}_eval.json"

if [[ ! -f "$GEN_OUT" || ! -s "$GEN_OUT" ]]; then
    echo "Generating for $STEP_NAME on $EVAL_TAG"
    python -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
        data.path="$EVAL_DATA" \
        data.prompt_key=prompt \
        data.n_samples=1 \
        data.output_path="$GEN_OUT" \
        model.path="$MERGED_DIR" \
        +model.trust_remote_code=True \
        rollout.temperature=0 \
        rollout.prompt_length="$PROMPT_LEN" \
        rollout.response_length="$RESP_LEN" \
        rollout.tensor_model_parallel_size=1 \
        rollout.gpu_memory_utilization=0.8
fi

echo "Running LLM-judge eval -> $EVAL_OUT"
python -m verl.eval.cli \
    --dataset simpleqa \
    --input "$GEN_OUT" \
    --output "$EVAL_OUT" \
    --use-judge

echo "Done: $EVAL_OUT"
