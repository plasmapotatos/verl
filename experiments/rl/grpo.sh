#!/usr/bin/env bash
set -e
set -x

export HYDRA_FULL_ERROR=1

VERL_DIR="${VERL_DIR:-/work/hdd/bbsg/twei2/rl/verl}"
cd "$VERL_DIR"

PROJECT_NAME=${PROJECT_NAME:-verl_examples}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-simpleqa_grpo}
TRAIN_DATA=${TRAIN_DATA:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/rich_sft/simpleqa_rich_sft_train_origqa.parquet}
VAL_DATA=${VAL_DATA:-$TRAIN_DATA}
EVAL_DATA=${EVAL_DATA:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/rich_sft/simpleqa_rich_sft_train_frac0.1_origqa.parquet}
SFT_DIR=${SFT_DIR:-}
if [[ -n "$SFT_DIR" && -z "${MODEL_PATH:-}" ]]; then
	LATEST_SFT_CKPT=$(find "$SFT_DIR" -maxdepth 1 -type d -name "global_step_*" | sort -V | tail -n 1)
	if [[ -z "$LATEST_SFT_CKPT" ]]; then
		echo "ERROR: no global_step_* checkpoint found in SFT_DIR=$SFT_DIR" >&2
		exit 1
	fi
	if [[ -d "$LATEST_SFT_CKPT/merged_hf_model" && -n "$(ls -A "$LATEST_SFT_CKPT/merged_hf_model" 2>/dev/null)" ]]; then
		MODEL_PATH="$LATEST_SFT_CKPT/merged_hf_model"
	else
		echo "ERROR: $LATEST_SFT_CKPT/merged_hf_model does not exist or is empty" >&2
		echo "Available contents: $(ls "$LATEST_SFT_CKPT")" >&2
		exit 1
	fi
	echo "Resolved MODEL_PATH from SFT_DIR: $MODEL_PATH"
fi
MODEL_PATH=${MODEL_PATH:-/work/hdd/bbsg/twei2/rl/verl/outputs/simpleqa_rich_sft/sft_lr1.5e-4_ep10_seed1/global_step_370/merged_hf_model}
RUN_EVAL=${RUN_EVAL:-1}
SKIP_TRAIN=${SKIP_TRAIN:-0}
USE_JUDGE=${USE_JUDGE:-1}
PRUNE_CHECKPOINTS=${PRUNE_CHECKPOINTS:-1}
PASS_AT_K_DATASET=${PASS_AT_K_DATASET:-simpleqa}
PASS_AT_K_EVAL_DATA=${PASS_AT_K_EVAL_DATA:-}
PASS_AT_K_TOP_K=${PASS_AT_K_TOP_K:-32}
PASS_AT_K_TOP_P=${PASS_AT_K_TOP_P:-0.9}
PASS_AT_K_TEMPERATURE=${PASS_AT_K_TEMPERATURE:-1}
RUN_BASE_EVAL=${RUN_BASE_EVAL:-1}
RUN_BASE_PASS_AT_K=${RUN_BASE_PASS_AT_K:-1}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}
PROMPT_LEN=${PROMPT_LEN:-1024}
RESP_LEN=${RESP_LEN:-256}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}
ROLLOUT_NAME=${ROLLOUT_NAME:-vllm}
ROLLOUT_N=${ROLLOUT_N:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
SAVE_FREQ=${SAVE_FREQ:-200}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
NNODES=${NNODES:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
REWARD_MODE=${REWARD_MODE:-binary}

CKPT_ROOT="outputs/rl/$PROJECT_NAME/$EXPERIMENT_NAME"
PLOT_DIR=${PLOT_DIR:-"$CKPT_ROOT/plots"}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"$CKPT_ROOT/rollouts"}
BASE_DIR="${CKPT_ROOT}/base"
STEP0_DIR="${CKPT_ROOT}/global_step_0"

pass_at_k_eval_tag() {
	local eval_data="$1"
	local eval_name
	eval_name="$(basename "${eval_data%.parquet}")"
	printf '%s\n' "$eval_name" | sed 's/[^A-Za-z0-9._-]/_/g'
}

pass_at_k_dataset_dir() {
	local output_root="$1"
	local eval_data="$2"
	local eval_tag
	eval_tag="$(pass_at_k_eval_tag "$eval_data")"
	printf '%s\n' "$output_root/pass@k/$eval_tag"
}

pass_at_k_dataset_done() {
	local output_root="$1"
	local eval_data="$2"
	local target_dir
	target_dir="$(pass_at_k_dataset_dir "$output_root" "$eval_data")"
	[[ -f "$target_dir/pass_at_k.json" ]]
}

finalize_pass_at_k_dataset_outputs() {
	local output_root="$1"
	local eval_data="$2"
	local top_k="$3"
	local target_dir
	target_dir="$(pass_at_k_dataset_dir "$output_root" "$eval_data")"
	mkdir -p "$target_dir"

	local src_gen="$output_root/pass@k/generations_${top_k}.parquet"
	local src_eval="$output_root/pass@k/eval_${top_k}.json"
	local src_summary="$output_root/pass@k/pass_at_k_${top_k}.json"

	[[ -f "$src_gen" ]] && mv "$src_gen" "$target_dir/generations.parquet"
	[[ -f "$src_eval" ]] && mv "$src_eval" "$target_dir/eval.json"
	[[ -f "$src_summary" ]] && mv "$src_summary" "$target_dir/pass_at_k.json"
}

grpo_train() {
	if [[ "$SKIP_TRAIN" == "1" ]]; then
		return 0
	fi

	# Calculate expected total steps: total_epochs * (len(train_data) / batch_size)
	NUM_SAMPLES=$(python3 -c "
import pandas as pd
df = pd.read_parquet('$TRAIN_DATA')
print(len(df))
")
	EXPECTED_TOTAL_STEPS=$(( TOTAL_EPOCHS * (NUM_SAMPLES / TRAIN_BATCH_SIZE) ))

	# Check if training is already complete
	LATEST_CKPT=$(find "$CKPT_ROOT" -maxdepth 1 -type d -name "global_step_*" | sort -V | tail -n 1)
	if [[ -n "$LATEST_CKPT" ]]; then
		LATEST_STEP=$(basename "$LATEST_CKPT" | sed 's/global_step_//')
		if [[ "$LATEST_STEP" -ge "$EXPECTED_TOTAL_STEPS" ]]; then
			echo "Training already complete (step $LATEST_STEP >= $EXPECTED_TOTAL_STEPS), skipping training."
			return 0
		fi
	fi

	CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python3 -m verl.trainer.main_ppo \
		algorithm.adv_estimator=grpo \
		data.train_files="$TRAIN_DATA" \
		data.val_files="$VAL_DATA" \
		data.train_batch_size="$TRAIN_BATCH_SIZE" \
		data.prompt_key=prompt \
		data.max_prompt_length="$PROMPT_LEN" \
		data.max_response_length="$RESP_LEN" \
		data.filter_overlong_prompts=True \
		data.truncation='error' \
		actor_rollout_ref.model.path="$MODEL_PATH" \
		actor_rollout_ref.actor.optim.lr=1e-6 \
		actor_rollout_ref.model.use_remove_padding=False \
		actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
		actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
		actor_rollout_ref.actor.use_kl_loss=True \
		actor_rollout_ref.actor.kl_loss_coef=0.01 \
		actor_rollout_ref.actor.kl_loss_type=low_var_kl \
		actor_rollout_ref.actor.entropy_coeff=0 \
		actor_rollout_ref.model.enable_gradient_checkpointing=True \
		actor_rollout_ref.actor.fsdp_config.param_offload=False \
		actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
		actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
		actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
		actor_rollout_ref.rollout.name="$ROLLOUT_NAME" \
		actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
		actor_rollout_ref.rollout.enable_chunked_prefill=False \
		actor_rollout_ref.rollout.enforce_eager=False \
		actor_rollout_ref.rollout.free_cache_engine=True \
		actor_rollout_ref.rollout.n="$ROLLOUT_N" \
		actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
		actor_rollout_ref.ref.fsdp_config.param_offload=True \
		algorithm.use_kl_in_reward=False \
		trainer.critic_warmup=0 \
		trainer.logger='["console","wandb"]' \
		trainer.project_name="$PROJECT_NAME" \
		trainer.experiment_name="$EXPERIMENT_NAME" \
		trainer.default_local_dir="$CKPT_ROOT" \
		trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
		trainer.resume_mode=auto \
		trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
		trainer.nnodes="$NNODES" \
		trainer.save_freq="$SAVE_FREQ" \
		trainer.test_freq="$SAVE_FREQ" \
		trainer.total_epochs="$TOTAL_EPOCHS" \
		trainer.esi_redundant_time=300 \
		reward_model.reward_mode="$REWARD_MODE"
}

grpo_eval() {
	if [[ "$RUN_EVAL" != "1" ]]; then
		return 0
	fi
	export MODEL_PATH
	export EXPERIMENT_NAME
	bash /work/hdd/bbsg/twei2/rl/verl/experiments/utils/eval_all_checkpoints.sh \
		"$CKPT_ROOT" \
		"$EVAL_DATA" \
		"" \
		"$N_GPUS_PER_NODE" \
		"$PROMPT_LEN" \
		"$RESP_LEN" \
		"$USE_JUDGE"
}

grpo_pass_at_k() {
	if [[ -z "$PASS_AT_K_EVAL_DATA" ]]; then
		return 0
	fi

	local pass_eval_items
	pass_eval_items="${PASS_AT_K_EVAL_DATA//,/ }"

	local eval_data
	for eval_data in $pass_eval_items; do
		[[ -n "$eval_data" ]] || continue
		if [[ ! -f "$eval_data" ]]; then
			echo "Skipping pass@k dataset (missing file): $eval_data"
			continue
		fi

		python3 "$VERL_DIR/scripts/run_pass_at_k_experiment.py" \
			--experiment-dir "$CKPT_ROOT" \
			--dataset "$PASS_AT_K_DATASET" \
			--eval-data "$eval_data" \
			--ks "$PASS_AT_K_TOP_K" \
			--layout dataset_subdir \
			--output-dir "$CKPT_ROOT" \
			--pass-at-k-args \
			--top-p "$PASS_AT_K_TOP_P" \
			--temperature "$PASS_AT_K_TEMPERATURE" \
			--prompt-len "$PROMPT_LEN" \
			--resp-len "$RESP_LEN" \
			--n-gpus "$N_GPUS_PER_NODE" \
			--use-judge
	done
}

grpo_step0_pass_at_k() {
	if [[ -z "$PASS_AT_K_EVAL_DATA" ]]; then
		return 0
	fi

	local pass_eval_items
	pass_eval_items="${PASS_AT_K_EVAL_DATA//,/ }"

	mkdir -p "$STEP0_DIR"

	local eval_data
	for eval_data in $pass_eval_items; do
		[[ -n "$eval_data" ]] || continue
		if [[ ! -f "$eval_data" ]]; then
			echo "Skipping step-0 pass@k dataset (missing file): $eval_data"
			continue
		fi
		if pass_at_k_dataset_done "$STEP0_DIR" "$eval_data"; then
			echo "Skipping step-0 pass@k dataset (already exists): $eval_data"
			continue
		fi

		python3 "$VERL_DIR/scripts/pass_at_k.py" \
			--checkpoint "$MODEL_PATH" \
			--dataset "$PASS_AT_K_DATASET" \
			--eval-data "$eval_data" \
			--output-dir "$STEP0_DIR" \
			--top-k "$PASS_AT_K_TOP_K" \
			--top-p "$PASS_AT_K_TOP_P" \
			--temperature "$PASS_AT_K_TEMPERATURE" \
			--prompt-len "$PROMPT_LEN" \
			--resp-len "$RESP_LEN" \
			--n-gpus "$N_GPUS_PER_NODE" \
			--use-judge

		finalize_pass_at_k_dataset_outputs "$STEP0_DIR" "$eval_data" "$PASS_AT_K_TOP_K"
	done
}

grpo_base_eval() {
	if [[ "$RUN_BASE_EVAL" != "1" ]]; then
		return 0
	fi

	if [[ -z "$EVAL_DATA" ]]; then
		return 0
	fi

	local gen_out_dir="$BASE_DIR/generations"
	mkdir -p "$gen_out_dir"

	local eval_path
	for eval_path in $EVAL_DATA; do
		[[ -n "$eval_path" ]] || continue
		if [[ ! -f "$eval_path" ]]; then
			echo "Skipping base eval dataset (missing file): $eval_path"
			continue
		fi

		local eval_tag
		eval_tag="eval_$(basename "${eval_path%.parquet}")"
		local gen_out_eval="$gen_out_dir/base_global_step_0__on_${eval_tag}.parquet"

		if [[ ! -f "$gen_out_eval" || ! -s "$gen_out_eval" ]]; then
			echo "Base generation -> $gen_out_eval"
			python3 -m verl.trainer.main_generation \
				trainer.nnodes=1 \
				trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
				data.path="$eval_path" \
				data.prompt_key=prompt \
				data.n_samples=1 \
				data.output_path="$gen_out_eval" \
				model.path="$BASE_MODEL_PATH" \
				+model.trust_remote_code=True \
				rollout.temperature=0 \
				rollout.prompt_length="$PROMPT_LEN" \
				rollout.response_length="$RESP_LEN" \
				rollout.tensor_model_parallel_size=1 \
				rollout.gpu_memory_utilization=0.8
		fi

		if [[ -f "$gen_out_eval" && -s "$gen_out_eval" ]]; then
			local eval_out
			eval_out="${gen_out_eval%.parquet}_eval.json"
			if [[ ! -f "$eval_out" ]]; then
				if [[ "$USE_JUDGE" == "1" ]]; then
					python3 -m verl.eval.cli \
						--dataset "$PASS_AT_K_DATASET" \
						--input "$gen_out_eval" \
						--output "$eval_out" \
						--use-judge
				else
					python3 -m verl.eval.cli \
						--dataset "$PASS_AT_K_DATASET" \
						--input "$gen_out_eval" \
						--output "$eval_out"
				fi
			fi
		fi
	done
}

grpo_base_pass_at_k() {
	if [[ "$RUN_BASE_PASS_AT_K" != "1" || -z "$PASS_AT_K_EVAL_DATA" ]]; then
		return 0
	fi

	local pass_eval_items
	pass_eval_items="${PASS_AT_K_EVAL_DATA//,/ }"

	mkdir -p "$BASE_DIR"

	local eval_data
	for eval_data in $pass_eval_items; do
		[[ -n "$eval_data" ]] || continue
		if [[ ! -f "$eval_data" ]]; then
			echo "Skipping base pass@k dataset (missing file): $eval_data"
			continue
		fi
		if pass_at_k_dataset_done "$BASE_DIR" "$eval_data"; then
			echo "Skipping base pass@k dataset (already exists): $eval_data"
			continue
		fi

		python3 "$VERL_DIR/scripts/pass_at_k.py" \
			--checkpoint "$BASE_MODEL_PATH" \
			--dataset "$PASS_AT_K_DATASET" \
			--eval-data "$eval_data" \
			--output-dir "$BASE_DIR" \
			--top-k "$PASS_AT_K_TOP_K" \
			--top-p "$PASS_AT_K_TOP_P" \
			--temperature "$PASS_AT_K_TEMPERATURE" \
			--prompt-len "$PROMPT_LEN" \
			--resp-len "$RESP_LEN" \
			--n-gpus "$N_GPUS_PER_NODE" \
			--use-judge

		finalize_pass_at_k_dataset_outputs "$BASE_DIR" "$eval_data" "$PASS_AT_K_TOP_K"
	done
}

grpo_plot() {
	if [[ ! -d "$CKPT_ROOT" ]]; then
		return 0
	fi

	mkdir -p "$PLOT_DIR"
	python3 "$VERL_DIR/scripts/plot_sft_eval_metrics.py" \
		--project-dir "$CKPT_ROOT" \
		--output-dir "$PLOT_DIR"
}

grpo_prune() {
	if [[ "$PRUNE_CHECKPOINTS" != "1" ]]; then
		return 0
	fi
	echo "Pruning checkpoints in $CKPT_ROOT"
	python3 "$VERL_DIR/scripts/prune_experiment_checkpoints.py" "$CKPT_ROOT"
}

grpo_run() {
	grpo_train
	grpo_eval
	grpo_pass_at_k
	grpo_step0_pass_at_k
	grpo_base_eval
	grpo_base_pass_at_k
	grpo_plot
	grpo_prune
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
	grpo_run
	# Signal chain_job.sh that this script has completed all its work
	[[ -n "${CHAIN_FLAG_FILE:-}" ]] && touch "$CHAIN_FLAG_FILE"
fi
