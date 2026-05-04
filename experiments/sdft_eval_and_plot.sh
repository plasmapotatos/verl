#!/usr/bin/env bash
set -euo pipefail

# Evaluate all SDFT checkpoints on train_eval + val, then plot.
#
# SDFT checkpoints are HF model dirs: checkpoint-{step}/
# The plotter expects: {exp_name}/global_step_{step}/generations/*_eval.json
#
# This script creates symlinked global_step_* dirs so the plotter works.

# make sure we DON'T join some external ray cluster
unset RAY_ADDRESS
unset RAY_REDIS_ADDRESS
unset RAY_HEAD_IP
unset RAY_PORT
export RAY_DISABLE_DASHBOARD=1
export RAY_USAGE_STATS_ENABLED=0

VERL_DIR="${VERL_DIR:-/work/hdd/bbsg/twei2/rl/verl}"

usage() {
    cat <<EOF
Usage: $0 <sdft_dir> <eval_file> [<eval_file> ...]
   or: SDFT_DIR=... EVAL_DATA_FILES="f1 f2" $0

Arguments:
  sdft_dir    Experiment directory containing checkpoint-* subdirs
  eval_file   One or more .parquet eval files

Env overrides: VERL_DIR, EXP_NAME, NGPU_GEN, PROMPT_LEN, RESP_LEN, USE_JUDGE
EOF
    exit 1
}

if [[ $# -ge 2 ]]; then
    SDFT_DIR="$1"
    shift
    EVAL_DATA_FILES="$*"
elif [[ $# -eq 1 ]]; then
    usage
fi

SDFT_DIR="${SDFT_DIR:?SDFT_DIR required (pass as first arg or env var)}"
EVAL_DATA_FILES="${EVAL_DATA_FILES:?EVAL_DATA_FILES required (pass as args or env var)}"
EXP_NAME="${EXP_NAME:-$(basename "$SDFT_DIR")}"
NGPU_GEN="${NGPU_GEN:-4}"
PROMPT_LEN="${PROMPT_LEN:-2048}"
RESP_LEN="${RESP_LEN:-1024}"
USE_JUDGE="${USE_JUDGE:-1}"

# Create a plot-compatible layout dir
PLOT_ROOT="$SDFT_DIR/eval_layout/$EXP_NAME"
mkdir -p "$PLOT_ROOT"

cd "$VERL_DIR"

echo "SDFT_DIR=$SDFT_DIR"
echo "EXP_NAME=$EXP_NAME"
echo "EVAL_DATA_FILES=$EVAL_DATA_FILES"
echo "PLOT_ROOT=$PLOT_ROOT"
echo ""

# Build (model_path, step) pairs to evaluate. If BASE_MODEL is set, eval it as step 0.
EVAL_PAIRS=()
if [[ -n "${BASE_MODEL:-}" ]]; then
    EVAL_PAIRS+=("0|${BASE_MODEL}")
fi
while IFS= read -r ckpt_dir; do
    [[ -z "$ckpt_dir" ]] && continue
    step="${ckpt_dir##*/checkpoint-}"
    EVAL_PAIRS+=("${step}|${ckpt_dir}")
done < <(find "$SDFT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -t'-' -k2 -n)

for pair in "${EVAL_PAIRS[@]}"; do
    step="${pair%%|*}"
    ckpt_dir="${pair#*|}"
    ckpt_name="$(basename "$ckpt_dir")"

    echo "=============================="
    echo "Processing $ckpt_name (step=$step)"
    echo "=============================="

    # Create global_step_* layout for the plotter
    step_dir="$PLOT_ROOT/global_step_${step}"
    gen_dir="$step_dir/generations"
    mkdir -p "$gen_dir"

    for eval_path in $EVAL_DATA_FILES; do
        eval_tag="$(basename "${eval_path%.parquet}")"
        gen_out="$gen_dir/${EXP_NAME}_global_step_${step}__on_eval_${eval_tag}.parquet"
        eval_out="${gen_out%.parquet}_eval.json"

        if [[ -f "$eval_out" ]]; then
            echo "Evaluation already exists: $eval_out (skipping)"
            continue
        fi

        # Generate predictions
        if [[ ! -f "$gen_out" || ! -s "$gen_out" ]]; then
            echo "Generating for step $step on $eval_tag -> $gen_out"
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

        # Evaluate
        if [[ -f "$gen_out" && -s "$gen_out" ]]; then
            echo "Grading step $step on $eval_tag -> $eval_out"
            if [[ "$USE_JUDGE" == "1" ]]; then
                python3 -m verl.eval.cli \
                    --dataset simpleqa \
                    --input "$gen_out" \
                    --output "$eval_out" \
                    --use-judge
            else
                python3 -m verl.eval.cli \
                    --dataset simpleqa \
                    --input "$gen_out" \
                    --output "$eval_out"
            fi
        else
            echo "Generation failed or empty for step $step on $eval_tag, skipping eval."
        fi
    done
done

# Plot
PLOT_DIR="$SDFT_DIR/plots"
echo ""
echo "Plotting -> $PLOT_DIR"
python3 "$VERL_DIR/scripts/plot_sft_eval_metrics.py" \
    --project-dir "$SDFT_DIR/eval_layout" \
    --output-dir "$PLOT_DIR"

echo ""
echo "DONE."
echo "Eval results: $PLOT_ROOT/global_step_*/generations/*_eval.json"
echo "Plots: $PLOT_DIR"
