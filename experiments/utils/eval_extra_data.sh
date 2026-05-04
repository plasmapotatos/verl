#!/usr/bin/env bash
set -e

# Run eval on additional eval datasets for an existing experiment.
# Generates and evaluates each checkpoint (global_step_*) under the experiment
# dir on each provided eval parquet, writing into the existing generations/
# folders using the same naming convention as experiments/rl/grpo.sh, then
# refreshes the plots/ directory.
#
# Usage:
#   eval_extra_data.sh <experiment_dir> <eval_parquet> [eval_parquet ...]
#
# Optional env vars:
#   DATASET           dataset name passed to verl.eval.cli (default: simpleqa)
#   N_GPUS_PER_NODE   default 4
#   PROMPT_LEN        default 1024
#   RESP_LEN          default 256
#   USE_JUDGE         default 1
#   MODEL_PATH        SFT/base model used for global_step_0 (skip if unset)
#   BASE_MODEL_PATH   pretrained base used for base/ dir (skip if unset)
#   PLOT_DIR          override plot output (default: <experiment_dir>/plots)

EXPERIMENT_DIR="$1"
shift || true

if [[ -z "$EXPERIMENT_DIR" || $# -eq 0 ]]; then
	echo "Usage: $0 <experiment_dir> <eval_parquet> [eval_parquet ...]" >&2
	exit 1
fi
if [[ ! -d "$EXPERIMENT_DIR" ]]; then
	echo "ERROR: experiment dir not found: $EXPERIMENT_DIR" >&2
	exit 1
fi

EVAL_DATA_LIST=("$@")

EXPERIMENT_NAME="$(basename "$EXPERIMENT_DIR")"
DATASET="${DATASET:-simpleqa}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
RESP_LEN="${RESP_LEN:-256}"
USE_JUDGE="${USE_JUDGE:-1}"
PLOT_DIR="${PLOT_DIR:-$EXPERIMENT_DIR/plots}"
VERL_DIR="${VERL_DIR:-/work/hdd/bbsg/twei2/rl/verl}"

run_eval_on_gen() {
	local gen_out="$1"
	local eval_out="${gen_out%.parquet}_eval.json"
	if [[ -f "$eval_out" ]]; then
		echo "  eval exists: $(basename "$eval_out")"
		return 0
	fi
	if [[ ! -f "$gen_out" || ! -s "$gen_out" ]]; then
		echo "  generation missing/empty, skipping eval: $gen_out"
		return 0
	fi
	if [[ "$USE_JUDGE" == "1" ]]; then
		python3 -m verl.eval.cli --dataset "$DATASET" --input "$gen_out" --output "$eval_out" --use-judge
	else
		python3 -m verl.eval.cli --dataset "$DATASET" --input "$gen_out" --output "$eval_out"
	fi
}

generate() {
	local model_dir="$1"
	local eval_path="$2"
	local gen_out="$3"
	if [[ -f "$gen_out" && -s "$gen_out" ]]; then
		echo "  generation exists: $(basename "$gen_out")"
		return 0
	fi
	echo "  generating -> $gen_out"
	python3 -m verl.trainer.main_generation \
		trainer.nnodes=1 \
		trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
		data.path="$eval_path" \
		data.prompt_key=prompt \
		data.n_samples=1 \
		data.output_path="$gen_out" \
		model.path="$model_dir" \
		+model.trust_remote_code=True \
		rollout.temperature=0 \
		rollout.prompt_length="$PROMPT_LEN" \
		rollout.response_length="$RESP_LEN" \
		rollout.tensor_model_parallel_size=1 \
		rollout.gpu_memory_utilization=0.8
}

process_ckpt() {
	# $1 = ckpt_dir (e.g. .../global_step_100, .../global_step_0, .../base)
	# $2 = model_path to use for generation
	local ckpt_dir="$1"
	local model_dir="$2"
	local step_name
	step_name="$(basename "$ckpt_dir")"

	local gen_dir="$ckpt_dir/generations"
	mkdir -p "$gen_dir"

	local eval_path eval_tag gen_out gen_prefix
	if [[ "$step_name" == "base" ]]; then
		gen_prefix="base_global_step_0"
	else
		gen_prefix="${EXPERIMENT_NAME}_${step_name}"
	fi

	for eval_path in "${EVAL_DATA_LIST[@]}"; do
		if [[ ! -f "$eval_path" ]]; then
			echo "  skip missing eval data: $eval_path"
			continue
		fi
		if [[ "$step_name" == "base" ]]; then
			eval_tag="eval_$(basename "${eval_path%.parquet}")"
		else
			eval_tag="$(basename "${eval_path%.parquet}")"
		fi
		gen_out="$gen_dir/${gen_prefix}__on_${eval_tag}.parquet"

		echo "[$step_name] eval_data=$(basename "$eval_path")"
		generate "$model_dir" "$eval_path" "$gen_out"
		run_eval_on_gen "$gen_out"
	done
}

# Trained checkpoints (global_step_N, N>0): merge if needed, then run.
CKPT_DIRS=$(find "$EXPERIMENT_DIR" -maxdepth 1 -type d -name "global_step_*" | sort -V)
for ckpt_dir in $CKPT_DIRS; do
	step_name="$(basename "$ckpt_dir")"
	step_num="${step_name#global_step_}"
	if [[ "$step_num" == "0" ]]; then
		if [[ -z "${MODEL_PATH:-}" ]]; then
			echo "[global_step_0] MODEL_PATH not set, skipping."
			continue
		fi
		process_ckpt "$ckpt_dir" "$MODEL_PATH"
		continue
	fi
	merged="$ckpt_dir/merged_hf_model"
	bash "$VERL_DIR/experiments/utils/merge_checkpoint.sh" "$ckpt_dir"
	if [[ ! -f "$merged/config.json" ]]; then
		echo "[$step_name] merged_hf_model missing, skipping."
		continue
	fi
	process_ckpt "$ckpt_dir" "$merged"
done

# Base pretrained model dir, if present.
BASE_DIR_PATH="$EXPERIMENT_DIR/base"
if [[ -d "$BASE_DIR_PATH" ]]; then
	if [[ -n "${BASE_MODEL_PATH:-}" ]]; then
		process_ckpt "$BASE_DIR_PATH" "$BASE_MODEL_PATH"
	else
		echo "[base] BASE_MODEL_PATH not set, skipping."
	fi
fi

mkdir -p "$PLOT_DIR"
python3 "$VERL_DIR/scripts/plot_sft_eval_metrics.py" \
	--project-dir "$EXPERIMENT_DIR" \
	--output-dir "$PLOT_DIR"
