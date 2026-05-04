#!/usr/bin/env bash
set -e

# Evaluate all checkpoints in a project, or a single checkpoint
# Usage: eval_all_checkpoints.sh <ckpt_root> <eval_data> [ckpt_dir] [n_gpus_per_node] [prompt_len] [resp_len] [use_judge]

CKPT_ROOT="$1"
EVAL_DATA="$2"
CKPT_DIR_ARG="$3"
N_GPUS_PER_NODE="${4:-2}"
PROMPT_LEN="${5:-2048}"
RESP_LEN="${6:-512}"
USE_JUDGE="${7:-1}"

if [[ -z "$CKPT_ROOT" || -z "$EVAL_DATA" ]]; then
    echo "Usage: $0 <ckpt_root> <eval_data> [ckpt_dir] [n_gpus_per_node] [prompt_len] [resp_len] [use_judge]"
    exit 1
fi

if [[ -n "$CKPT_DIR_ARG" ]]; then
    CKPT_DIRS="$CKPT_DIR_ARG"
else
    STEP0_DIR="$CKPT_ROOT/global_step_0"
    mkdir -p "$STEP0_DIR/generations"
    CKPT_DIRS=$(find "$CKPT_ROOT" -maxdepth 1 -type d -name "global_step_*" | sort -V)
    if [[ -z "$CKPT_DIRS" ]]; then
        CKPT_DIRS="$STEP0_DIR"
    fi
fi

for CKPT_DIR in $CKPT_DIRS; do
    STEP_NAME=$(basename "$CKPT_DIR")
    
    if [[ "$STEP_NAME" == "global_step_0" ]]; then
        MERGED_DIR="$MODEL_PATH"
    else
        MERGED_DIR="$CKPT_DIR/merged_hf_model"
        bash /work/hdd/bbsg/twei2/rl/verl/experiments/utils/merge_checkpoint.sh "$CKPT_DIR"
        if [[ ! -f "$MERGED_DIR/config.json" ]]; then
            echo "merged_hf_model missing or invalid for $STEP_NAME, skipping."
            continue
        fi
    fi

    GEN_DIR="$CKPT_DIR/generations"
    mkdir -p "$GEN_DIR"

    # Generate and evaluate for each eval data item
    for EVAL_DATA_ITEM in $EVAL_DATA; do
        EVAL_TAG="$(basename "${EVAL_DATA_ITEM%.parquet}")"
        GEN_OUT="$GEN_DIR/${EXPERIMENT_NAME}_${STEP_NAME}__on_${EVAL_TAG}.parquet"
        EVAL_OUT="${GEN_OUT%.parquet}_eval.json"

        if [[ -f "$EVAL_OUT" ]]; then
            echo "Evaluation already exists for $STEP_NAME on $EVAL_TAG, skipping."
            continue
        fi

        if [[ ! -f "$GEN_OUT" || ! -s "$GEN_OUT" ]]; then
            echo "Generating for $STEP_NAME on $EVAL_TAG"
            python3 -m verl.trainer.main_generation \
                trainer.nnodes=1 \
                trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
                data.path="$EVAL_DATA_ITEM" \
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

        if [[ -f "$GEN_OUT" && -s "$GEN_OUT" ]]; then
            if [[ "$USE_JUDGE" == "1" ]]; then
                python3 -m verl.eval.cli \
                    --dataset simpleqa \
                    --input "$GEN_OUT" \
                    --output "$EVAL_OUT" \
                    --use-judge
            else
                python3 -m verl.eval.cli \
                    --dataset simpleqa \
                    --input "$GEN_OUT" \
                    --output "$EVAL_OUT"
            fi
        else
            echo "Generation failed or empty for $STEP_NAME on $EVAL_TAG, skipping eval."
        fi
    done
done

# Plot only if evaluating all
if [[ -z "$CKPT_DIR_ARG" ]]; then
    PLOT_DIR="${PLOT_DIR:-$CKPT_ROOT/plots}"
    python3 /work/hdd/bbsg/twei2/rl/verl/scripts/plot_sft_eval_metrics.py \
        --project-dir "$CKPT_ROOT" \
        --output-dir "$PLOT_DIR"
fi
