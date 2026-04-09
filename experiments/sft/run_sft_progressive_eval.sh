#!/usr/bin/env bash
set -euo pipefail

# make sure we DON'T join some external ray cluster
unset RAY_ADDRESS
unset RAY_REDIS_ADDRESS
unset RAY_HEAD_IP
unset RAY_PORT
export RAY_DISABLE_DASHBOARD=1
export RAY_USAGE_STATS_ENABLED=0

# --------------------
# USAGE
# --------------------
# Required env vars (or edit defaults below):
#   PROJECT_NAME   e.g. "simpleqa_sft_aug"
#   TRAIN_DATA     path to parquet used for SFT training
#   EVAL_DATA      space-separated paths to parquet used for eval (optional)
#
# Optional:
#   EXP_PREFIX     prefix for experiment_name (default: "sft")
#   EPOCHS_LIST    space-separated epochs (default: "1 3 6 10 20")
#   LR_LIST        space-separated learning rates (default: "1e-5 2e-5 3e-5 5e-5 7e-5 1e-4 1.5e-4 2e-4 3e-4 5e-4")
#   CLM_MODE       set to 1 for text-only CLM training (default: 0)
#   CLM_TEXT_KEY   text field name for CLM (default: "text")
#   CLM_MAX_LEN    max sequence length for CLM (default: 4096)
#   CLM_TRUNCATION truncation mode for CLM (default: "right")
#   TRAIN_BATCH_SIZE (default: 64)
#   SEED           (default: 1)
#   NPROC          (default: 1)
#   NGPU_GEN       (default: 1)
#   TP_SIZE        (default: 1)
#   TEMP           (default: 0)
#   MAX_LENGTH     max sequence length (default: 1024)
#   TRUNCATION     truncation mode: error|left|right (default: error)
#   FILTER_OVERLONG_PROMPTS filter samples exceeding max_length (default: 0)
#   PROMPT_KEY     top-level prompt field (default: "prompt")
#   RESPONSE_KEY   top-level response field (default: "response")
#   PROMPT_DICT_KEYS   dict path under prompt field (default: "question")
#   RESPONSE_DICT_KEYS dict path under response field (default: "answer")
#   PROMPT_LEN     (default: 2048)
#   RESP_LEN       (default: 1024)
#   GPU_MEM_UTIL   (default: 0.8)
#   RUN_BASE_EVAL  run base Qwen eval before sweeps (default: 1)
#   PASS_AT_K_MODE pass@k mode: none|last|all (default: all)
#   PASS_AT_K_TOP_K (default: 32)
#   PASS_AT_K_TOP_P (default: 0.9)
#   PASS_AT_K_TEMPERATURE (default: 1)
#   PASS_AT_K_DATASET (default: "simpleqa")
#   PASS_AT_K_EVAL_DATA single parquet for pass@k (required to run pass@k)
#   POLL_INTERVAL polling interval in seconds (default: 15)
#
# Example:
#   PROJECT_NAME=simpleqa_sft_llm_direct \
#   TRAIN_DATA=/.../llm_paraphrase_train_direct.parquet \
#   EVAL_DATA=/.../base/train.parquet \
#   bash run_sft_progressive_eval.sh
# --------------------

# --------------------
# CONFIG (EDIT THESE)
# --------------------
VERL_DIR="${VERL_DIR:-/work/hdd/bbsg/twei2/rl/verl}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

PROJECT_NAME="${PROJECT_NAME:-}"
TRAIN_DATA="${TRAIN_DATA:-}"
EVAL_DATA="${EVAL_DATA:-}"

EXP_PREFIX="${EXP_PREFIX:-sft}"
EPOCHS_LIST="${EPOCHS_LIST:-1 3 6 10 20}"
LR_LIST="${LR_LIST:-3e-5 5e-5 7e-5 1e-4 1.5e-4 2e-4}"
CLM_MODE="${CLM_MODE:-0}"
CLM_TEXT_KEY="${CLM_TEXT_KEY:-text}"
CLM_MAX_LEN="${CLM_MAX_LEN:-4096}"
CLM_TRUNCATION="${CLM_TRUNCATION:-right}"

# Training launcher settings
NPROC="${NPROC:-1}"

# Fixed hyperparams
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
SEED="${SEED:-1}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"

# Generation settings
N_SAMPLES="${N_SAMPLES:-1}"
NGPU_GEN="${NGPU_GEN:-1}"
TP_SIZE="${TP_SIZE:-1}"
TEMP="${TEMP:-0}"
PROMPT_LEN="${PROMPT_LEN:-2048}"
RESP_LEN="${RESP_LEN:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
RUN_BASE_EVAL="${RUN_BASE_EVAL:-1}"
PASS_AT_K_MODE="${PASS_AT_K_MODE:-all}"
PASS_AT_K_TOP_K="${PASS_AT_K_TOP_K:-32}"
PASS_AT_K_TOP_P="${PASS_AT_K_TOP_P:-0.9}"
PASS_AT_K_TEMPERATURE="${PASS_AT_K_TEMPERATURE:-1}"
PASS_AT_K_DATASET="${PASS_AT_K_DATASET:-simpleqa}"
PASS_AT_K_EVAL_DATA="${PASS_AT_K_EVAL_DATA:-}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"

if [[ -n "${RUN_PASS_AT_K:-}" ]]; then
	if [[ "$RUN_PASS_AT_K" == "0" ]]; then
		PASS_AT_K_MODE="none"
	fi
fi

# Sequence settings
MAX_LENGTH="${MAX_LENGTH:-1024}"
TRUNCATION="${TRUNCATION:-error}"
FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-0}"

# SFT field mapping (non-CLM)
PROMPT_KEY="${PROMPT_KEY:-prompt}"
RESPONSE_KEY="${RESPONSE_KEY:-response}"
PROMPT_DICT_KEYS="${PROMPT_DICT_KEYS-}"
RESPONSE_DICT_KEYS="${RESPONSE_DICT_KEYS-}"


# --------------------
# Validate required inputs
# --------------------
if [[ -z "$PROJECT_NAME" ]]; then
	echo "ERROR: PROJECT_NAME is required"
	exit 1
fi
if [[ -z "$TRAIN_DATA" ]]; then
	echo "ERROR: TRAIN_DATA is required (path to parquet used for SFT)"
	exit 1
fi
if [[ ! -f "$TRAIN_DATA" ]]; then
	echo "ERROR: TRAIN_DATA not found: $TRAIN_DATA"
	exit 1
fi
if [[ -n "$EVAL_DATA" ]]; then
	for eval_path in $EVAL_DATA; do
		if [[ ! -f "$eval_path" ]]; then
			echo "ERROR: EVAL_DATA not found: $eval_path"
			exit 1
		fi
	done
fi

# Output layout: outputs/sft/{project_name}/{experiment_name}
ROOT="$VERL_DIR/outputs/sft/$PROJECT_NAME"
mkdir -p "$ROOT"
cd "$VERL_DIR"

echo "VERL_DIR=$VERL_DIR"
echo "PROJECT_NAME=$PROJECT_NAME"
echo "ROOT=$ROOT"
echo "GEN_OUT_DIR=per-step (outputs/sft/$PROJECT_NAME/{experiment_name}/global_step_*/generations)"
echo "TRAIN_DATA=$TRAIN_DATA"
if [[ -n "$EVAL_DATA" ]]; then
	echo "EVAL_DATA=$EVAL_DATA"
else
	echo "EVAL_DATA not provided; eval generation/grade will be skipped"
fi
echo "EPOCHS_LIST=$EPOCHS_LIST"
echo "LR_LIST=$LR_LIST"
echo "CLM_MODE=$CLM_MODE"
if [[ "$CLM_MODE" == "1" ]]; then
	echo "CLM_TEXT_KEY=$CLM_TEXT_KEY"
	echo "CLM_MAX_LEN=$CLM_MAX_LEN"
	echo "CLM_TRUNCATION=$CLM_TRUNCATION"
fi
echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
echo "SEED=$SEED"
echo "RUN_BASE_EVAL=$RUN_BASE_EVAL"
echo "PASS_AT_K_MODE=$PASS_AT_K_MODE"
echo "POLL_INTERVAL=$POLL_INTERVAL"
if [[ "$PASS_AT_K_MODE" != "none" ]]; then
	echo "PASS_AT_K_TOP_K=$PASS_AT_K_TOP_K"
	echo "PASS_AT_K_TOP_P=$PASS_AT_K_TOP_P"
	echo "PASS_AT_K_TEMPERATURE=$PASS_AT_K_TEMPERATURE"
	echo "PASS_AT_K_DATASET=$PASS_AT_K_DATASET"
	echo "PASS_AT_K_EVAL_DATA=${PASS_AT_K_EVAL_DATA:-<unset>}"
fi
if [[ "$CLM_MODE" != "1" ]]; then
	echo "PROMPT_KEY=$PROMPT_KEY"
	echo "RESPONSE_KEY=$RESPONSE_KEY"
	if [[ -n "$PROMPT_DICT_KEYS" ]]; then
		echo "PROMPT_DICT_KEYS=$PROMPT_DICT_KEYS"
	else
		echo "PROMPT_DICT_KEYS=(none)"
	fi
	if [[ -n "$RESPONSE_DICT_KEYS" ]]; then
		echo "RESPONSE_DICT_KEYS=$RESPONSE_DICT_KEYS"
	else
		echo "RESPONSE_DICT_KEYS=(none)"
	fi
	echo "MAX_LENGTH=$MAX_LENGTH"
	echo "TRUNCATION=$TRUNCATION"
	echo "FILTER_OVERLONG_PROMPTS=$FILTER_OVERLONG_PROMPTS"
fi

afail () {
	echo "ERROR: $1"
	exit 1
}

sync_hf_aux_files () {
	local merged_dir="$1"
	mkdir -p "$merged_dir"
	if [[ "$BASE_MODEL" == *"VL"* ]]; then
		local aux_dir="${HF_AUX_SRC_DIR:-$VERL_DIR/configs/hf_aux_files}"
		local preproc_src="$aux_dir/preprocessor_config.json"
		local chat_template_src="$aux_dir/chat_template.json"
		if [[ -f "$preproc_src" ]]; then
			cp -f "$preproc_src" "$merged_dir/preprocessor_config.json"
		else
			echo "WARNING: missing $preproc_src (not copied)"
		fi
		if [[ -f "$chat_template_src" ]]; then
			cp -f "$chat_template_src" "$merged_dir/chat_template.json"
		else
			echo "WARNING: missing $chat_template_src (not copied)"
		fi
	fi
}

pick_last_ckpt () {
	local exp_dir="$1"
	find "$exp_dir" -maxdepth 2 -type d -name "global_step_*" | sort -V | tail -n 1
}

get_num_rows_in_parquet () {
	local parquet_path="$1"
	python3 - "$parquet_path" <<'PY'
import sys
import pyarrow.parquet as pq
path = sys.argv[1]
print(pq.ParquetFile(path).metadata.num_rows)
PY
}

get_max_epoch () {
	local max_ep=0
	for ep in $EPOCHS_LIST; do
		if (( ep > max_ep )); then
			max_ep="$ep"
		fi
	done
	echo "$max_ep"
}

get_steps_per_epoch () {
	local num_rows
	num_rows="$(get_num_rows_in_parquet "$TRAIN_DATA")"
	if [[ -z "$num_rows" ]]; then
		afail "Failed to determine num_rows for $TRAIN_DATA"
	fi
	python3 - "$num_rows" "$TRAIN_BATCH_SIZE" <<'PY'
import math, sys
num_rows = int(sys.argv[1])
batch = int(sys.argv[2])
print(math.ceil(num_rows / batch))
PY
}

build_target_epoch_step_map () {
	local steps_per_epoch="$1"
	local out_file="$2"
	: > "$out_file"
	for ep in $EPOCHS_LIST; do
		local target_step=$(( ep * steps_per_epoch ))
		echo "$ep:$target_step" >> "$out_file"
	done
}

ckpt_step_from_dir () {
	local ckpt_dir="$1"
	local base
	base="$(basename "$ckpt_dir")"
	echo "${base#global_step_}"
}

update_candidate_checkpoints () {
	local exp_dir="$1"
	local target_map_file="$2"
	local candidates_file="$3"

	declare -A candidates=()
	if [[ -f "$candidates_file" ]]; then
		while IFS=: read -r step dir; do
			[[ -n "$step" && -n "$dir" ]] || continue
			candidates["$step"]="$dir"
		done < "$candidates_file"
	fi

	mapfile -t target_steps < <(awk -F: '{print $2}' "$target_map_file")
	mapfile -t sorted_ckpts < <(printf '%s\n' "$exp_dir"/global_step_* 2>/dev/null | sort -V)

	if [[ ${#sorted_ckpts[@]} -eq 0 ]]; then
		return
	fi

	for target_step in "${target_steps[@]}"; do
		[[ -n "$target_step" ]] || continue
		if [[ -n "${candidates[$target_step]:-}" ]]; then
			continue
		fi
		for ckpt_dir in "${sorted_ckpts[@]}"; do
			local step
			step="$(ckpt_step_from_dir "$ckpt_dir")"
			if [[ "$step" =~ ^[0-9]+$ ]] && (( step >= target_step )); then
				candidates["$target_step"]="$ckpt_dir"
				break
			fi
		done
	done

	: > "$candidates_file"
	for target_step in "${target_steps[@]}"; do
		if [[ -n "${candidates[$target_step]:-}" ]]; then
			echo "$target_step:${candidates[$target_step]}" >> "$candidates_file"
		fi
	done

	local latest_ckpt
	latest_ckpt="${sorted_ckpts[-1]}"
	for ckpt_dir in "${sorted_ckpts[@]}"; do
		local keep=0
		if [[ "$ckpt_dir" == "$latest_ckpt" ]]; then
			keep=1
		else
			for target_step in "${target_steps[@]}"; do
				if [[ "$ckpt_dir" == "${candidates[$target_step]:-}" ]]; then
					keep=1
					break
				fi
			done
		fi
		if [[ "$keep" -eq 0 ]]; then
			echo "Pruning checkpoint (non-eval): $ckpt_dir"
			rm -rf "$ckpt_dir"
		fi
	done
}

prune_checkpoints () {
	local exp_dir="$1"
	local keep_file="$2"
	local max_step="$3"

	if [[ ! -f "$keep_file" ]]; then
		return
	fi

	mapfile -t keep_dirs < "$keep_file"
	for ckpt_dir in "$exp_dir"/global_step_*; do
		[[ -d "$ckpt_dir" ]] || continue
		local base
		base="$(basename "$ckpt_dir")"
		local step
		step="${base#global_step_}"
		if [[ "$step" =~ ^[0-9]+$ ]] && (( step > max_step )); then
			continue
		fi
		local keep=0
		for keep_dir in "${keep_dirs[@]}"; do
			if [[ "$ckpt_dir" == "$keep_dir" ]]; then
				keep=1
				break
			fi
		done
		if [[ "$keep" -eq 0 ]]; then
			echo "Pruning checkpoint: $ckpt_dir"
			rm -rf "$ckpt_dir"
		fi
	done
}

run_pass_at_k () {
	local exp_name="$1"
	local exp_dir="$2"
	if [[ "$PASS_AT_K_MODE" == "none" ]]; then
		return
	fi
	if [[ "$PASS_AT_K_MODE" != "last" && "$PASS_AT_K_MODE" != "all" ]]; then
		afail "PASS_AT_K_MODE must be one of: none|last|all"
	fi
	read -r -a pass_at_k_paths <<<"${PASS_AT_K_EVAL_DATA:-}"
	if [[ ${#pass_at_k_paths[@]} -eq 0 ]]; then
		echo "Skipping pass@k (PASS_AT_K_EVAL_DATA not provided)"
		return
	fi

	local run_for_ckpt out_dir eval_data eval_tag merged_dir step

	if [[ "$PASS_AT_K_MODE" == "last" ]]; then
		local ckpt_dir
		ckpt_dir="$(pick_last_ckpt "$exp_dir")"
		if [[ -z "$ckpt_dir" ]]; then
			echo "Skipping $exp_name (no global_step_* found)"
			return
		fi
		merged_dir="$ckpt_dir/merged_hf_model"
		if [[ ! -d "$merged_dir" ]]; then
			echo "Skipping $exp_name (missing merged_hf_model under $ckpt_dir)"
			return
		fi

		for eval_data in "${pass_at_k_paths[@]}"; do
			eval_tag="passatk_$(basename "${eval_data%.parquet}")"
			out_dir="$exp_dir/pass_at_k/$eval_tag/last"
			mkdir -p "$out_dir"
			echo "Running pass@k for $exp_name @ last -> $merged_dir (data=$eval_data)"
			python3 "$VERL_DIR/scripts/pass_at_k.py" \
				--checkpoint "$merged_dir" \
				--dataset "$PASS_AT_K_DATASET" \
				--eval-data "$eval_data" \
				--output-dir "$out_dir" \
				--top-k "$PASS_AT_K_TOP_K" \
				--top-p "$PASS_AT_K_TOP_P" \
				--temperature "$PASS_AT_K_TEMPERATURE" \
				--prompt-key "${PROMPT_KEY:-prompt}" \
				--use-judge
		done
		return
	fi

	local ckpt_dirs
	ckpt_dirs=("$exp_dir"/global_step_*)
	if [[ ${#ckpt_dirs[@]} -eq 0 ]]; then
		echo "Skipping $exp_name (no global_step_* found)"
		return
	fi
	local sorted_ckpts
	sorted_ckpts=( $(printf '%s\n' "${ckpt_dirs[@]}" | sort -V) )
	for ckpt_dir in "${sorted_ckpts[@]}"; do
		merged_dir="$ckpt_dir/merged_hf_model"
		if [[ ! -d "$merged_dir" ]]; then
			echo "Skipping $exp_name @ $ckpt_dir (missing merged_hf_model)"
			continue
		fi
		step="$(basename "$ckpt_dir")"
		for eval_data in "${pass_at_k_paths[@]}"; do
			eval_tag="passatk_$(basename "${eval_data%.parquet}")"
			out_dir="$exp_dir/$step/pass_at_k/$eval_tag"
			mkdir -p "$out_dir"
			echo "Running pass@k for $exp_name @ $step -> $merged_dir (data=$eval_data)"
			python3 "$VERL_DIR/scripts/pass_at_k.py" \
				--checkpoint "$merged_dir" \
				--dataset "$PASS_AT_K_DATASET" \
				--eval-data "$eval_data" \
				--output-dir "$out_dir" \
				--top-k "$PASS_AT_K_TOP_K" \
				--top-p "$PASS_AT_K_TOP_P" \
				--temperature "$PASS_AT_K_TEMPERATURE" \
				--prompt-key "${PROMPT_KEY:-prompt}" \
				--use-judge
		done
		done
}

run_generation () {
	local tag="$1" # "train" or "eval"
	local data_path="$2"
	local merged_dir="$3"
	local out_path="$4"

		if [[ -s "$out_path" ]]; then
		echo "Generation already exists: $out_path (skipping)"
		return
	fi

	echo "Generating ($tag) -> $out_path"
	python3 -m verl.trainer.main_generation \
		trainer.nnodes=1 \
		trainer.n_gpus_per_node="$NGPU_GEN" \
		data.path="$data_path" \
		data.prompt_key="${PROMPT_KEY:-prompt}" \
		data.n_samples="$N_SAMPLES" \
		data.output_path="$out_path" \
		model.path="$merged_dir" \
		+model.trust_remote_code=True \
		rollout.temperature="$TEMP" \
		rollout.prompt_length="$PROMPT_LEN" \
		rollout.response_length="$RESP_LEN" \
		rollout.tensor_model_parallel_size="$TP_SIZE" \
		rollout.gpu_memory_utilization="$GPU_MEM_UTIL"

	echo "Wrote $out_path"
}

run_grade () {
	local parquet_path="$1"
	local out_json="${parquet_path%.parquet}_eval.json"

	if [[ -f "$out_json" ]]; then
		echo "Graded output already exists: $out_json (skipping)"
		return
	fi

	echo "Grading -> $out_json"
	python3 -m verl.eval.cli \
		--dataset simpleqa \
		--input "$parquet_path" \
		--output "$out_json" \
		--use-judge
}

run_eval_for_ckpt () {
	local exp_name="$1"
	local ckpt_dir="$2"

	echo ""
	echo "--- MERGE/EVAL: $exp_name @ $ckpt_dir"

	export EXPERIMENT_NAME="$exp_name"
	bash "$VERL_DIR/experiments/utils/merge_checkpoint.sh" "$ckpt_dir"

	if [[ -n "$EVAL_DATA" ]]; then
		bash "$VERL_DIR/experiments/utils/eval_all_checkpoints.sh" "$ROOT/$exp_name" "$EVAL_DATA" "$ckpt_dir" "$NGPU_GEN" "$PROMPT_LEN" "$RESP_LEN"
	else
		echo "Skipping eval generation/grade (EVAL_DATA not provided)"
	fi
}

run_base_eval () {
	if [[ -z "$EVAL_DATA" ]]; then
		echo "Skipping base eval (EVAL_DATA not provided)"
		return
	fi
	if [[ "$RUN_BASE_EVAL" != "1" ]]; then
		echo "Skipping base eval (RUN_BASE_EVAL=$RUN_BASE_EVAL)"
		return
	fi

	local base_dir="$ROOT/base"
	local gen_out_dir="$base_dir/generations"
	mkdir -p "$gen_out_dir"

	for eval_path in $EVAL_DATA; do
		local eval_tag
		eval_tag="eval_$(basename "${eval_path%.parquet}")"
		local gen_out_eval="$gen_out_dir/base_global_step_0__on_${eval_tag}.parquet"
			if [[ -s "$gen_out_eval" ]]; then
			echo "Base generation already exists: $gen_out_eval (skipping)"
		else
			echo "Base generation (Qwen) -> $gen_out_eval"
			python3 -m verl.trainer.main_generation \
				trainer.nnodes=1 \
				trainer.n_gpus_per_node="$NGPU_GEN" \
				data.path="$eval_path" \
				data.prompt_key="${PROMPT_KEY:-prompt}" \
				data.n_samples="$N_SAMPLES" \
				data.output_path="$gen_out_eval" \
				model.path="$BASE_MODEL" \
				+model.trust_remote_code=True \
				rollout.temperature="$TEMP" \
				rollout.prompt_length="$PROMPT_LEN" \
				rollout.response_length="$RESP_LEN" \
				rollout.tensor_model_parallel_size="$TP_SIZE" \
				rollout.gpu_memory_utilization="$GPU_MEM_UTIL"
		fi
		run_grade "$gen_out_eval"
	done

	if [[ "$PASS_AT_K_MODE" != "none" ]]; then
		read -r -a pass_at_k_paths <<<"${PASS_AT_K_EVAL_DATA:-}"
		if [[ ${#pass_at_k_paths[@]} -gt 0 ]]; then
			local base_passatk_root="$base_dir/pass_at_k"
			for eval_path in "${pass_at_k_paths[@]}"; do
				local eval_tag
				eval_tag="passatk_$(basename "${eval_path%.parquet}")"
				local out_dir="$base_passatk_root/$eval_tag"
				mkdir -p "$out_dir"
				echo "Running pass@k for base model -> $out_dir (data=$eval_path)"
				python3 "$VERL_DIR/scripts/pass_at_k.py" \
					--checkpoint "$BASE_MODEL" \
					--dataset "$PASS_AT_K_DATASET" \
					--eval-data "$eval_path" \
					--output-dir "$out_dir" \
					--top-k "$PASS_AT_K_TOP_K" \
					--top-p "$PASS_AT_K_TOP_P" \
					--temperature "$PASS_AT_K_TEMPERATURE" \
					--prompt-key "${PROMPT_KEY:-prompt}" \
					--use-judge
			done
		fi
	fi
}

launch_train_background () {
	local exp_name="$1"
	local max_epoch="$2"
	local out_dir="$ROOT/$exp_name"
	mkdir -p "$out_dir"

	echo ""
	echo "=============================="
	echo "TRAIN (progressive): $exp_name (epochs=$max_epoch, lr=$LR)"
	echo "OUT: $out_dir"
	echo "DATA: $TRAIN_DATA"
	echo "=============================="

	# shellcheck disable=SC2086
	torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
		-m verl.trainer.fsdp_sft_trainer \
		data.train_batch_size="$TRAIN_BATCH_SIZE" \
		data.train_files="$TRAIN_DATA" \
		data.val_files="$TRAIN_DATA" \
		optim.lr="$LR" \
		data.micro_batch_size=4 \
		model.partial_pretrain="$BASE_MODEL" \
		trainer.default_local_dir="$out_dir" \
		trainer.project_name="$PROJECT_NAME" \
		trainer.experiment_name="$exp_name" \
		trainer.logger=[console,wandb] \
		trainer.total_epochs="$max_epoch" \
		trainer.resume_mode=auto \
		trainer.save_freq=100 \
		trainer.seed="$SEED" \
		model.fsdp_config.model_dtype="$MODEL_DTYPE" \
		ulysses_sequence_parallel_size=1 \
		use_remove_padding=true \
		$(
			if [[ "$CLM_MODE" == "1" ]]; then
				echo "data.text_key=$CLM_TEXT_KEY data.max_length=$CLM_MAX_LEN data.truncation=$CLM_TRUNCATION"
			else
				args="data.prompt_key=$PROMPT_KEY data.response_key=$RESPONSE_KEY data.max_length=$MAX_LENGTH data.truncation=$TRUNCATION +data.filter_overlong_prompts=$FILTER_OVERLONG_PROMPTS"
				if [[ -n "$PROMPT_DICT_KEYS" ]]; then
					args+=" data.prompt_dict_keys=['$PROMPT_DICT_KEYS']"
				fi
				if [[ -n "$RESPONSE_DICT_KEYS" ]]; then
					args+=" +data.response_dict_keys=['$RESPONSE_DICT_KEYS']"
				fi
				echo "$args"
			fi
		) &

	echo $! > "$out_dir/train.pid"
}

refresh_plots () {
	if [[ -n "$EVAL_DATA" ]]; then
		echo "Refreshing eval metrics -> $ROOT/plots"
		python3 "$VERL_DIR/scripts/plot_sft_eval_metrics.py" \
			--project-dir "$ROOT" \
			--output-dir "$ROOT/plots"
	fi
}

maybe_eval_target_epoch () {
	local exp_name="$1"
	local exp_dir="$2"
	local target_ep="$3"
	local target_step="$4"
	local processed_steps_file="$5"
	local evaled_epochs_file="$6"
	local evaled_ckpts_file="$7"

	# If this epoch already evaluated, skip
	if grep -qx "${target_ep}" "$evaled_epochs_file" 2>/dev/null; then
		return
	fi

	# Mode A: exact epoch-step matching
	local exact_dir="$exp_dir/global_step_${target_step}"
	if [[ -d "$exact_dir" ]]; then
		if ! grep -qx "${exact_dir}" "$processed_steps_file" 2>/dev/null; then
			echo "Exact match for epoch $target_ep -> $exact_dir"
			run_eval_for_ckpt "$exp_name" "$exact_dir"
			echo "$exact_dir" >> "$processed_steps_file"
			echo "$target_ep" >> "$evaled_epochs_file"
			echo "$exact_dir" >> "$evaled_ckpts_file"
			run_pass_at_k "$exp_name" "$exp_dir"
			refresh_plots
			prune_checkpoints "$exp_dir" "$evaled_ckpts_file" "$target_step"
		fi
		return
	fi

	# Mode B: nearest checkpoint fallback
	# Find the first checkpoint with step >= target_step
	local candidate_dir=""
	local ckpt_dirs
	ckpt_dirs=("$exp_dir"/global_step_*)
	if [[ ${#ckpt_dirs[@]} -gt 0 ]]; then
		local sorted_ckpts
		sorted_ckpts=( $(printf '%s\n' "${ckpt_dirs[@]}" | sort -V) )
		for ckpt_dir in "${sorted_ckpts[@]}"; do
			local step
			step="${ckpt_dir##*/}"
			step="${step#global_step_}"
			if [[ "$step" =~ ^[0-9]+$ ]] && (( step >= target_step )); then
				candidate_dir="$ckpt_dir"
				break
			fi
		done
	fi

	if [[ -n "$candidate_dir" ]]; then
		if ! grep -qx "${candidate_dir}" "$processed_steps_file" 2>/dev/null; then
			echo "Fallback match for epoch $target_ep -> $candidate_dir"
			run_eval_for_ckpt "$exp_name" "$candidate_dir"
			echo "$candidate_dir" >> "$processed_steps_file"
			echo "$target_ep" >> "$evaled_epochs_file"
			echo "$candidate_dir" >> "$evaled_ckpts_file"
			run_pass_at_k "$exp_name" "$exp_dir"
			refresh_plots
			local candidate_step
			candidate_step="${candidate_dir##*/}"
			candidate_step="${candidate_step#global_step_}"
			if [[ "$candidate_step" =~ ^[0-9]+$ ]]; then
				prune_checkpoints "$exp_dir" "$evaled_ckpts_file" "$candidate_step"
			else
				prune_checkpoints "$exp_dir" "$evaled_ckpts_file" "$target_step"
			fi
		fi
	fi
}

run_latest_checkpoint_if_needed () {
	local exp_name="$1"
	local exp_dir="$2"
	local processed_steps_file="$3"
	local evaled_epochs_file="$4"
	local evaled_ckpts_file="$5"

	local latest_ckpt
	latest_ckpt="$(pick_last_ckpt "$exp_dir")"
	if [[ -z "$latest_ckpt" ]]; then
		echo "No checkpoints available for $exp_name; skipping final eval."
		return
	fi

	if grep -qx "$latest_ckpt" "$processed_steps_file" 2>/dev/null; then
		echo "Latest checkpoint already evaluated: $latest_ckpt"
		return
	fi

	echo "Evaluating final checkpoint for $exp_name -> $latest_ckpt"
	run_eval_for_ckpt "$exp_name" "$latest_ckpt"
	echo "$latest_ckpt" >> "$processed_steps_file"
	echo "latest" >> "$evaled_epochs_file"
	echo "$latest_ckpt" >> "$evaled_ckpts_file"
	run_pass_at_k "$exp_name" "$exp_dir"
	refresh_plots

	local latest_step
	latest_step="$(ckpt_step_from_dir "$latest_ckpt")"
	if [[ "$latest_step" =~ ^[0-9]+$ ]]; then
		prune_checkpoints "$exp_dir" "$evaled_ckpts_file" "$latest_step"
	else
		prune_checkpoints "$exp_dir" "$evaled_ckpts_file" 0
	fi
}

monitor_and_eval_checkpoints () {
	local exp_name="$1"
	local exp_dir="$2"
	local target_map_file="$3"
	local candidates_file="$4"
	local train_pid="$5"

	# Poll until training finishes
	while kill -0 "$train_pid" 2>/dev/null; do
		update_candidate_checkpoints "$exp_dir" "$target_map_file" "$candidates_file"
		sleep "$POLL_INTERVAL"
	done
}

run_progressive_lr () {
	local max_epoch
	max_epoch="$(get_max_epoch)"
	if [[ -z "$max_epoch" || "$max_epoch" == "0" ]]; then
		afail "Failed to compute max epoch from EPOCHS_LIST"
	fi

	local steps_per_epoch
	steps_per_epoch="$(get_steps_per_epoch)"
	if [[ -z "$steps_per_epoch" || "$steps_per_epoch" == "0" ]]; then
		afail "Failed to compute steps per epoch"
	fi

	local exp_name="${EXP_PREFIX}_lr${LR}_epmax${max_epoch}_seed${SEED}"
	local exp_dir="$ROOT/$exp_name"
	mkdir -p "$exp_dir"

	local expected_total_steps=$(( max_epoch * steps_per_epoch ))
	local latest_ckpt
	latest_ckpt="$(pick_last_ckpt "$exp_dir")"
	local latest_step
	local skip_training=0
	if [[ -n "$latest_ckpt" ]]; then
		latest_step="$(ckpt_step_from_dir "$latest_ckpt")"
		if [[ "$latest_step" =~ ^[0-9]+$ ]] && (( latest_step >= expected_total_steps )); then
			skip_training=1
			echo "Training already complete for $exp_name (step=$latest_step >= $expected_total_steps), skipping training."
		fi
	fi

	local target_map_file="$exp_dir/target_epoch_step_map.txt"
	local processed_steps_file="$exp_dir/processed_checkpoints.txt"
	local evaled_epochs_file="$exp_dir/evaluated_epochs.txt"
	local evaled_ckpts_file="$exp_dir/evaluated_checkpoints.txt"
	local candidates_file="$exp_dir/eval_candidates.txt"

	build_target_epoch_step_map "$steps_per_epoch" "$target_map_file"
	: > "$processed_steps_file"
	: > "$evaled_epochs_file"
	: > "$evaled_ckpts_file"
	: > "$candidates_file"

	echo "Target epochs -> steps (steps_per_epoch=$steps_per_epoch):"
	cat "$target_map_file"

	if [[ "$skip_training" -eq 0 ]]; then
		launch_train_background "$exp_name" "$max_epoch"
		local train_pid
		train_pid="$(cat "$exp_dir/train.pid")"

		monitor_and_eval_checkpoints "$exp_name" "$exp_dir" "$target_map_file" \
			"$candidates_file" "$train_pid"

		# Wait for training to fully finish
		wait "$train_pid"
	else
		echo "Skipping progressive training for $exp_name; using existing checkpoints for evaluation."
	fi

	# Final pass in case last checkpoint appeared near the end
	while IFS=: read -r ep target_step; do
		maybe_eval_target_epoch "$exp_name" "$exp_dir" "$ep" "$target_step" \
			"$processed_steps_file" "$evaled_epochs_file" "$evaled_ckpts_file"
	done < "$target_map_file"
	run_latest_checkpoint_if_needed "$exp_name" "$exp_dir" \
		"$processed_steps_file" "$evaled_epochs_file" "$evaled_ckpts_file"
}



echo ""
echo "==== SFT PROGRESSIVE EVAL (project=$PROJECT_NAME) ===="

run_base_eval

for lr in $LR_LIST; do
	LR="$lr"
	run_progressive_lr

done


echo ""
echo "DONE."
echo "Training outputs: $ROOT/{experiment_name}/global_step_*"
echo "Generation outputs: $ROOT/{experiment_name}/generations"

if [[ -n "$EVAL_DATA" ]]; then
	echo ""
	echo "Plotting eval metrics -> $ROOT/plots"
	python3 "$VERL_DIR/scripts/plot_sft_eval_metrics.py" \
		--project-dir "$ROOT" \
		--output-dir "$ROOT/plots"
fi

# Signal chain_job.sh that this script has completed all its work
[[ -n "${CHAIN_FLAG_FILE:-}" ]] && touch "$CHAIN_FLAG_FILE"
