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

# Sequence settings
MAX_LENGTH="${MAX_LENGTH:-1024}"
TRUNCATION="${TRUNCATION:-error}"
FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-0}"

# SFT field mapping (non-CLM)
PROMPT_KEY="${PROMPT_KEY:-prompt}"
RESPONSE_KEY="${RESPONSE_KEY:-response}"
PROMPT_DICT_KEYS="${PROMPT_DICT_KEYS-}"
RESPONSE_DICT_KEYS="${RESPONSE_DICT_KEYS-}"

# HF aux files to copy into merged_hf_model (same as your inference script)
HF_AUX_SRC_DIR="${HF_AUX_SRC_DIR:-$VERL_DIR/configs/hf_aux_files}"
PREPROC_SRC="$HF_AUX_SRC_DIR/preprocessor_config.json"
CHAT_TEMPLATE_SRC="$HF_AUX_SRC_DIR/chat_template.json"

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

# Output layout: outputs/{project_name}/{experiment_name}
ROOT="$VERL_DIR/outputs/$PROJECT_NAME"
mkdir -p "$ROOT"
cd "$VERL_DIR"

echo "VERL_DIR=$VERL_DIR"
echo "PROJECT_NAME=$PROJECT_NAME"
echo "ROOT=$ROOT"
echo "GEN_OUT_DIR=per-experiment (outputs/$PROJECT_NAME/{experiment_name}/generations)"
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
echo "HF_AUX_SRC_DIR=$HF_AUX_SRC_DIR"
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

	if [[ -f "$PREPROC_SRC" ]]; then
		cp -f "$PREPROC_SRC" "$merged_dir/preprocessor_config.json"
	else
		echo "WARNING: missing $PREPROC_SRC (not copied)"
	fi

	if [[ -f "$CHAT_TEMPLATE_SRC" ]]; then
		cp -f "$CHAT_TEMPLATE_SRC" "$merged_dir/chat_template.json"
	else
		echo "WARNING: missing $CHAT_TEMPLATE_SRC (not copied)"
	fi
}

pick_last_ckpt () {
	local exp_dir="$1"
	find "$exp_dir" -maxdepth 2 -type d -name "global_step_*" | sort -V | tail -n 1
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
		model.partial_pretrain=Qwen/Qwen2.5-VL-3B-Instruct \
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
	local gen_out_dir="$ROOT/$exp_name/generations"
	mkdir -p "$gen_out_dir"
	local step
	step="$(basename "$ckpt_dir")"
	local merged_dir="$ckpt_dir/merged_hf_model"

	echo ""
	echo "--- MERGE/EVAL: $exp_name @ $ckpt_dir"

	# 1) Merge FSDP -> HF (skip if already merged)
	if [[ -d "$merged_dir" ]]; then
		echo "Merged model already exists: $merged_dir (skipping merge)"
	else
		echo "Merging -> $merged_dir"
		python3 -m verl.model_merger merge \
			--backend fsdp \
			--local_dir "$ckpt_dir" \
			--target_dir "$merged_dir"
	fi

	# 1.5) Ensure aux HF files are present in merged dir
	echo "Syncing HF aux files into: $merged_dir"
	sync_hf_aux_files "$merged_dir"

	# 2) Generation on EVAL_DATA (optional)
	if [[ -n "$EVAL_DATA" ]]; then
		for eval_path in $EVAL_DATA; do
			local eval_tag
			eval_tag="eval_$(basename "${eval_path%.parquet}")"
			local gen_out_eval="$gen_out_dir/${exp_name}_${step}__on_${eval_tag}.parquet"
			run_generation "eval" "$eval_path" "$merged_dir" "$gen_out_eval"
			run_grade "$gen_out_eval"
		done
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
}

echo ""
echo "==== SFT SWEEP + EVAL (project=$PROJECT_NAME) ===="

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