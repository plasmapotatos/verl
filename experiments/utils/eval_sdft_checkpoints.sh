#!/usr/bin/env bash
set -euo pipefail

# Evaluate all SDFT (HF-format) checkpoints under an experiment dir on one or
# more eval parquets, then lay the outputs out so plot_sft_eval_metrics.py
# can consume them.
#
# Layout produced:
#   <sdft_dir>/eval_layout/<exp_name>/global_step_<step>/generations/
#       <exp_name>_global_step_<step>__on_eval_<tag>.parquet
#       <exp_name>_global_step_<step>__on_eval_<tag>_eval.json
#
# Usage:
#   eval_sdft_checkpoints.sh <sdft_dir> <eval_parquet> [<eval_parquet> ...]
#
# Env overrides:
#   EXP_NAME (default: basename of sdft_dir)
#   NGPU_GEN (default: 4)
#   PROMPT_LEN (default: 2048)
#   RESP_LEN (default: 1024)
#   USE_JUDGE (default: 1)
#   DATASET (default: simpleqa)
#   MAKE_PLOTS (default: 1)

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <sdft_dir> <eval_parquet> [<eval_parquet> ...]" >&2
    exit 1
fi

SDFT_DIR="$(realpath "$1")"
shift
EVAL_DATA_FILES=("$@")

VERL_DIR="${VERL_DIR:-/work/hdd/bbsg/twei2/rl/verl}"
EXP_NAME="${EXP_NAME:-$(basename "$SDFT_DIR")}"
NGPU_GEN="${NGPU_GEN:-4}"
PROMPT_LEN="${PROMPT_LEN:-2048}"
RESP_LEN="${RESP_LEN:-1024}"
USE_JUDGE="${USE_JUDGE:-1}"
DATASET="${DATASET:-simpleqa}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"

# Do not inherit a foreign ray cluster.
unset RAY_ADDRESS RAY_REDIS_ADDRESS RAY_HEAD_IP RAY_PORT
export RAY_DISABLE_DASHBOARD=1
export RAY_USAGE_STATS_ENABLED=0

PLOT_ROOT="$SDFT_DIR/eval_layout/$EXP_NAME"
mkdir -p "$PLOT_ROOT"

cd "$VERL_DIR"

echo "SDFT_DIR=$SDFT_DIR"
echo "EXP_NAME=$EXP_NAME"
echo "EVAL_DATA_FILES=${EVAL_DATA_FILES[*]}"
echo "PLOT_ROOT=$PLOT_ROOT"
echo ""

ckpt_dirs=$(find "$SDFT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -t'-' -k2 -n)
if [[ -z "$ckpt_dirs" ]]; then
    echo "No checkpoint-* dirs found under $SDFT_DIR" >&2
    exit 1
fi

for ckpt_dir in $ckpt_dirs; do
    ckpt_name=$(basename "$ckpt_dir")
    step="${ckpt_name#checkpoint-}"

    if [[ ! -f "$ckpt_dir/config.json" ]]; then
        echo "Skipping $ckpt_name: no config.json (not an HF model dir)."
        continue
    fi

    echo "=============================="
    echo "Processing $ckpt_name (step=$step)"
    echo "=============================="

    step_dir="$PLOT_ROOT/global_step_${step}"
    gen_dir="$step_dir/generations"
    mkdir -p "$gen_dir"

    for eval_path in "${EVAL_DATA_FILES[@]}"; do
        if [[ ! -f "$eval_path" ]]; then
            echo "Eval parquet missing: $eval_path (skipping)"
            continue
        fi
        eval_tag="$(basename "${eval_path%.parquet}")"
        gen_out="$gen_dir/${EXP_NAME}_global_step_${step}__on_eval_${eval_tag}.parquet"
        eval_out="${gen_out%.parquet}_eval.json"

        if [[ -f "$eval_out" ]]; then
            echo "Eval already exists: $eval_out (skipping)"
            continue
        fi

        if [[ ! -f "$gen_out" || ! -s "$gen_out" ]]; then
            echo "Generating: step=$step eval=$eval_tag -> $gen_out"
            python3 -m verl.trainer.main_generation \
                trainer.nnodes=1 \
                trainer.n_gpus_per_node="$NGPU_GEN" \
                data.path="$eval_path" \
                data.prompt_key=prompt \
                data.n_samples=1 \
                data.output_path="$gen_out" \
                model.path="$ckpt_dir" \
                +model.trust_remote_code=True \
                rollout.temperature=0 \
                rollout.prompt_length="$PROMPT_LEN" \
                rollout.response_length="$RESP_LEN" \
                rollout.tensor_model_parallel_size=1 \
                rollout.gpu_memory_utilization=0.8
        fi

        if [[ -f "$gen_out" && -s "$gen_out" ]]; then
            echo "Grading: step=$step eval=$eval_tag -> $eval_out"
            judge_args=()
            if [[ "$USE_JUDGE" == "1" ]]; then
                judge_args+=(--use-judge)
            fi
            python3 -m verl.eval.cli \
                --dataset "$DATASET" \
                --input "$gen_out" \
                --output "$eval_out" \
                "${judge_args[@]}"
        else
            echo "Generation empty/missing for step=$step eval=$eval_tag, skipping eval."
        fi
    done
done

if [[ "$MAKE_PLOTS" == "1" ]]; then
    PLOT_DIR="${PLOT_DIR:-$SDFT_DIR/plots}"
    echo ""
    echo "Plotting -> $PLOT_DIR"
    python3 "$VERL_DIR/scripts/plot_sft_eval_metrics.py" \
        --project-dir "$SDFT_DIR/eval_layout" \
        --output-dir "$PLOT_DIR"
fi

echo ""
echo "DONE."
echo "Eval JSONs: $PLOT_ROOT/global_step_*/generations/*_eval.json"
