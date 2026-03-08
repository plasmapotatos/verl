#!/usr/bin/env bash
set -euo pipefail

PROJECT_PATH="outputs/simpleqa_rich_sft"
EVAL_DATA="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/rich_sft/simpleqa_rich_sft_train_frac0.1_origqa.parquet"
DATASET="simpleqa"
TOP_K=32
TOP_P=0.9
TEMPERATURE=0.8
PASS_AT_K_MODE="${PASS_AT_K_MODE:-last}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
shopt -s nullglob

if [[ ! -d "$PROJECT_PATH" ]]; then
	echo "ERROR: PROJECT_PATH not found: $PROJECT_PATH"
	exit 1
fi

if [[ ! -f "$EVAL_DATA" ]]; then
	echo "ERROR: eval parquet not found: $EVAL_DATA"
	exit 1
fi

if [[ "$PASS_AT_K_MODE" != "last" && "$PASS_AT_K_MODE" != "all" ]]; then
	echo "ERROR: PASS_AT_K_MODE must be one of: last|all"
	exit 1
fi

exp_dirs=("$PROJECT_PATH"/*)
sorted_exp_dirs=( $(printf '%s\n' "${exp_dirs[@]}" | sort -Vr) )

for exp_dir in "${sorted_exp_dirs[@]}"; do
	if [[ ! -d "$exp_dir" ]]; then
		continue
	fi


	exp_name="$(basename "$exp_dir")"

	ckpt_dirs=("$exp_dir"/global_step_*)
	if [[ ${#ckpt_dirs[@]} -eq 0 ]]; then
		echo "Skipping $exp_name (no global_step_* found)"
		continue
	fi

	if [[ "$PASS_AT_K_MODE" == "last" ]]; then
		last_ckpt="$(printf '%s\n' "${ckpt_dirs[@]}" | sort -V | tail -n 1)"
		if [[ -z "$last_ckpt" ]]; then
			echo "Skipping $exp_name (no global_step_* found)"
			continue
		fi

		merged_dir="$last_ckpt/merged_hf_model"
		if [[ ! -d "$merged_dir" ]]; then
			echo "Skipping $exp_name (missing merged_hf_model under $last_ckpt)"
			continue
		fi

		output_dir="$exp_dir"

		echo "Running pass@k for $exp_name -> $merged_dir"
		python3 "$REPO_ROOT/scripts/pass_at_k.py" \
			--checkpoint "$merged_dir" \
			--dataset "$DATASET" \
			--eval-data "$EVAL_DATA" \
			--output-dir "$output_dir" \
			--top-k "$TOP_K" \
			--top-p "$TOP_P" \
			--temperature "$TEMPERATURE" \
			--use-judge
		continue
	fi

	sorted_ckpts=( $(printf '%s\n' "${ckpt_dirs[@]}" | sort -V) )
	for ckpt_dir in "${sorted_ckpts[@]}"; do
		merged_dir="$ckpt_dir/merged_hf_model"
		if [[ ! -d "$merged_dir" ]]; then
			echo "Skipping $exp_name @ $ckpt_dir (missing merged_hf_model)"
			continue
		fi
		step="$(basename "$ckpt_dir")"
		output_dir="$exp_dir/$step"

		echo "Running pass@k for $exp_name @ $step -> $merged_dir"
		python3 "$REPO_ROOT/scripts/pass_at_k.py" \
			--checkpoint "$merged_dir" \
			--dataset "$DATASET" \
			--eval-data "$EVAL_DATA" \
			--output-dir "$output_dir" \
			--top-k "$TOP_K" \
			--top-p "$TOP_P" \
			--temperature "$TEMPERATURE" \
			--use-judge
	done
done
