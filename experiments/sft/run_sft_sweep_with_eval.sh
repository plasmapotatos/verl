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
#   PASS_AT_K_TEMPERATURE (default: 0.8)
#   PASS_AT_K_DATASET (default: "simpleqa")
#   PASS_AT_K_EVAL_DATA single parquet for pass@k (required to run pass@k)
#
# Example:
#   PROJECT_NAME=simpleqa_sft_llm_direct \
#   TRAIN_DATA=/.../llm_paraphrase_train_direct.parquet \
#   EVAL_DATA=/.../base/train.parquet \
#   bash run_sft_sweep_with_eval.sh
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
PASS_AT_K_TEMPERATURE="${PASS_AT_K_TEMPERATURE:-0.8}"
PASS_AT_K_DATASET="${PASS_AT_K_DATASET:-simpleqa}"
PASS_AT_K_EVAL_DATA="${PASS_AT_K_EVAL_DATA:-}"

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

run_pass_at_k () {
	local exp_name="$1"
	local exp_dir="$2"
	if [[ "$PASS_AT_K_MODE" == "none" ]]; then
		return
	fi
	if [[ "$PASS_AT_K_MODE" != "last" && "$PASS_AT_K_MODE" != "all" ]]; then
		echo "ERROR: PASS_AT_K_MODE must be one of: none|last|all"
		exit 1
	fi
	local eval_data="$PASS_AT_K_EVAL_DATA"
	if [[ -z "$eval_data" ]]; then
		echo "Skipping pass@k (PASS_AT_K_EVAL_DATA not provided)"
		return
	fi

	if [[ "$PASS_AT_K_MODE" == "last" ]]; then
		local ckpt_dir
		ckpt_dir="$(pick_last_ckpt "$exp_dir")"
		if [[ -z "$ckpt_dir" ]]; then
			echo "Skipping $exp_name (no global_step_* found)"
			return
		fi
		local merged_dir="$ckpt_dir/merged_hf_model"
		if [[ ! -d "$merged_dir" ]]; then
			echo "Skipping $exp_name (missing merged_hf_model under $ckpt_dir)"
			return
		fi

		echo "Running pass@k for $exp_name -> $merged_dir"
		python3 "$VERL_DIR/scripts/pass_at_k.py" \
			--checkpoint "$merged_dir" \
			--dataset "$PASS_AT_K_DATASET" \
			--eval-data "$eval_data" \
			--output-dir "$exp_dir" \
			--top-k "$PASS_AT_K_TOP_K" \
			--top-p "$PASS_AT_K_TOP_P" \
			--temperature "$PASS_AT_K_TEMPERATURE" \
			--use-judge
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
		local merged_dir="$ckpt_dir/merged_hf_model"
		if [[ ! -d "$merged_dir" ]]; then
			echo "Skipping $exp_name @ $ckpt_dir (missing merged_hf_model)"
			continue
		fi
		local step
		step="$(basename "$ckpt_dir")"
		local out_dir="$exp_dir/$step"
		echo "Running pass@k for $exp_name @ $step -> $merged_dir"
		python3 "$VERL_DIR/scripts/pass_at_k.py" \
			--checkpoint "$merged_dir" \
			--dataset "$PASS_AT_K_DATASET" \
			--eval-data "$eval_data" \
			--output-dir "$out_dir" \
			--top-k "$PASS_AT_K_TOP_K" \
			--top-p "$PASS_AT_K_TOP_P" \
			--temperature "$PASS_AT_K_TEMPERATURE" \
			--use-judge
	done
}

run_train () {
	local exp_name="$1"
	local epochs="$2"
	local out_dir="$ROOT/$exp_name"
	mkdir -p "$out_dir"

	echo ""
	echo "=============================="
	echo "TRAIN: $exp_name (epochs=$epochs, lr=$LR)"
	echo "OUT: $out_dir"
	echo "DATA: $TRAIN_DATA"
	echo "=============================="

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
		trainer.total_epochs="$epochs" \
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
		)
}

run_generation () {
	local tag="$1" # "train" or "eval"
	local data_path="$2"
	local merged_dir="$3"
	local out_path="$4"

	if [[ -f "$out_path" ]]; then
		echo "Generation already exists: $out_path (skipping)"
		return
	fi

	echo "Generating ($tag) -> $out_path"
	python3 -m verl.trainer.main_generation \
		trainer.nnodes=1 \
		trainer.n_gpus_per_node="$NGPU_GEN" \
		data.path="$data_path" \
		data.prompt_key=prompt \
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
		if [[ -f "$gen_out_eval" ]]; then
			echo "Base generation already exists: $gen_out_eval (skipping)"
		else
			echo "Base generation (Qwen) -> $gen_out_eval"
			python3 -m verl.trainer.main_generation \
				trainer.nnodes=1 \
				trainer.n_gpus_per_node="$NGPU_GEN" \
				data.path="$eval_path" \
				data.prompt_key=prompt \
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

	if [[ "$PASS_AT_K_MODE" != "none" && -n "$PASS_AT_K_EVAL_DATA" ]]; then
		echo "Running pass@k for base model"
		python3 "$VERL_DIR/scripts/pass_at_k.py" \
			--checkpoint "$BASE_MODEL" \
			--dataset "$PASS_AT_K_DATASET" \
			--eval-data "$PASS_AT_K_EVAL_DATA" \
			--output-dir "$base_dir" \
			--top-k "$PASS_AT_K_TOP_K" \
			--top-p "$PASS_AT_K_TOP_P" \
			--temperature "$PASS_AT_K_TEMPERATURE" \
			--use-judge
	fi
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

run_one () {
	local epochs="$1"
	local exp_name="${EXP_PREFIX}_lr${LR}_ep${epochs}_seed${SEED}"

	run_train "$exp_name" "$epochs"

	local exp_dir="$ROOT/$exp_name"
	local ckpt_dir
	ckpt_dir="$(pick_last_ckpt "$exp_dir")"

	if [[ -z "${ckpt_dir:-}" || ! -d "$ckpt_dir" ]]; then
		echo "ERROR: No global_step_* found under $exp_dir"
		exit 1
	fi

	run_eval_for_ckpt "$exp_name" "$ckpt_dir"
	run_pass_at_k "$exp_name" "$exp_dir"
}

echo ""
echo "==== SFT SWEEP + EVAL (project=$PROJECT_NAME) ===="

run_base_eval

for lr in $LR_LIST; do
	LR="$lr"
	for ep in $EPOCHS_LIST; do
		run_one "$ep"
	done
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