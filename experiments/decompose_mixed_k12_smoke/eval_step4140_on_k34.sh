#!/usr/bin/env bash
# Run the global_step_4140 SFT checkpoint on the k=3 and k=4 composition
# eval splits produced by build_train_eval_k34.sh.
#
# Mirrors the generation/eval naming convention used by
# experiments/utils/eval_extra_data.sh, but limited to a single checkpoint
# so we don't re-evaluate every global_step_* in the run dir.
#
# Generations + eval json are written under
#   <ckpt>/generations/<experiment>__global_step_4140__on_<eval_tag>.parquet
#   <ckpt>/generations/<experiment>__global_step_4140__on_<eval_tag>_eval.json

set -euo pipefail

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
SIF="/work/hdd/bbsg/twei2/rl/torch2501.sif"

EXPERIMENT_DIR="$WORKDIR/outputs/sft/simpleqa_decompose_mixed_k12_smoke_sft/sft_lr1.5e-4_epmax30_seed1"
CKPT_DIR="$EXPERIMENT_DIR/global_step_4140"
MERGED="$CKPT_DIR/merged_hf_model"
EXPERIMENT_NAME="$(basename "$EXPERIMENT_DIR")"
STEP_NAME="global_step_4140"

DATASET="${DATASET:-simpleqa}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
RESP_LEN="${RESP_LEN:-256}"
USE_JUDGE="${USE_JUDGE:-1}"

EVAL_DATA=(
    "$WORKDIR/data/simpleqa/partition/decompose_mixed_k12_smoke/sft/train_eval_k3.parquet"
    "$WORKDIR/data/simpleqa/partition/decompose_mixed_k12_smoke/sft/train_eval_k4.parquet"
)

if [[ ! -f "$MERGED/config.json" ]]; then
    echo "merged_hf_model not found; running merge_checkpoint.sh"
    bash "$WORKDIR/experiments/utils/merge_checkpoint.sh" "$CKPT_DIR"
fi
if [[ ! -f "$MERGED/config.json" ]]; then
    echo "ERROR: still no merged_hf_model at $MERGED" >&2
    exit 1
fi

GEN_DIR="$CKPT_DIR/generations"
mkdir -p "$GEN_DIR"
GEN_PREFIX="${EXPERIMENT_NAME}_${STEP_NAME}"

for eval_path in "${EVAL_DATA[@]}"; do
    if [[ ! -f "$eval_path" ]]; then
        echo "skip missing eval data: $eval_path"
        continue
    fi
    eval_tag="$(basename "${eval_path%.parquet}")"
    gen_out="$GEN_DIR/${GEN_PREFIX}__on_${eval_tag}.parquet"
    eval_out="${gen_out%.parquet}_eval.json"

    echo "[$STEP_NAME] eval_data=$(basename "$eval_path")"

    if [[ -f "$gen_out" && -s "$gen_out" ]]; then
        echo "  generation exists: $(basename "$gen_out")"
    else
        echo "  generating -> $gen_out"
        python -m verl.trainer.main_generation \
            trainer.nnodes=1 \
            trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
            data.path="$eval_path" \
            data.prompt_key=prompt \
            data.n_samples=1 \
            data.output_path="$gen_out" \
            model.path="$MERGED" \
            +model.trust_remote_code=True \
            rollout.temperature=0 \
            rollout.prompt_length="$PROMPT_LEN" \
            rollout.response_length="$RESP_LEN" \
            rollout.tensor_model_parallel_size=1 \
            rollout.gpu_memory_utilization=0.8
    fi

    if [[ -f "$eval_out" ]]; then
        echo "  eval exists: $(basename "$eval_out")"
        continue
    fi
    if [[ ! -f "$gen_out" || ! -s "$gen_out" ]]; then
        echo "  generation missing/empty, skipping eval"
        continue
    fi
    if [[ "$USE_JUDGE" == "1" ]]; then
        python -m verl.eval.cli \
            --dataset "$DATASET" --input "$gen_out" --output "$eval_out" --use-judge
    else
        python -m verl.eval.cli \
            --dataset "$DATASET" --input "$gen_out" --output "$eval_out"
    fi
done

echo "Done."
