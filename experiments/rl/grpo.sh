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
MODEL_PATH=${MODEL_PATH:-/work/hdd/bbsg/twei2/rl/verl/outputs/simpleqa_rich_sft/sft_lr1.5e-4_ep10_seed1/global_step_370/merged_hf_model}
RUN_EVAL=${RUN_EVAL:-1}
SKIP_TRAIN=${SKIP_TRAIN:-0}
USE_JUDGE=${USE_JUDGE:-1}
PRUNE_CHECKPOINTS=${PRUNE_CHECKPOINTS:-1}
PROMPT_LEN=${PROMPT_LEN:-1024}
RESP_LEN=${RESP_LEN:-256}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
ROLLOUT_NAME=${ROLLOUT_NAME:-vllm}
ROLLOUT_N=${ROLLOUT_N:-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
SAVE_FREQ=${SAVE_FREQ:-200}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
NNODES=${NNODES:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

CKPT_ROOT="outputs/rl/$PROJECT_NAME/$EXPERIMENT_NAME"
PLOT_DIR=${PLOT_DIR:-"$CKPT_ROOT/plots"}

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
		actor_rollout_ref.model.use_remove_padding=True \
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
		actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
		actor_rollout_ref.rollout.name="$ROLLOUT_NAME" \
		actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
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
		trainer.resume_mode=auto \
		trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
		trainer.nnodes="$NNODES" \
		trainer.save_freq="$SAVE_FREQ" \
		trainer.test_freq="$SAVE_FREQ" \
		trainer.total_epochs="$TOTAL_EPOCHS"
}

grpo_eval() {
	if [[ "$RUN_EVAL" != "1" ]]; then
		return 0
	fi
	bash /work/hdd/bbsg/twei2/rl/verl/experiments/utils/eval_all_checkpoints.sh \
		"$CKPT_ROOT" \
		"$EVAL_DATA" \
		"" \
		"$N_GPUS_PER_NODE" \
		"$PROMPT_LEN" \
		"$RESP_LEN" \
		"$USE_JUDGE"
}

grpo_prune() {
	if [[ "$PRUNE_CHECKPOINTS" != "1" ]]; then
		return 0
	fi
	echo "Pruning checkpoints in $CKPT_ROOT"
	python3 "$VERL_DIR/scripts/prune_checkpoints.py" "$CKPT_ROOT"
}

grpo_run() {
	grpo_train
	grpo_eval
	grpo_prune
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
	grpo_run
fi