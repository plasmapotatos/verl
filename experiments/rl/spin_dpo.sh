#!/usr/bin/env bash
set -e
set -x

export HYDRA_FULL_ERROR=1

VERL_DIR="${VERL_DIR:-/work/hdd/bbsg/twei2/rl/verl}"
cd "$VERL_DIR"

PROJECT_NAME=${PROJECT_NAME:-verl_examples}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-simpleqa_spin}
TRAIN_DATA=${TRAIN_DATA:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/rich_sft/simpleqa_rich_sft_train_origqa.parquet}
VAL_DATA=${VAL_DATA:-$TRAIN_DATA}
EVAL_DATA=${EVAL_DATA:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/rich_sft/simpleqa_rich_sft_train_frac0.1_origqa.parquet}
MODEL_PATH=${MODEL_PATH:-/work/hdd/bbsg/twei2/rl/verl/outputs/simpleqa_rich_sft/sft_lr1.5e-4_ep10_seed1/global_step_370/merged_hf_model}
REWARD_FN_PATH=${REWARD_FN_PATH:-/work/hdd/bbsg/twei2/rl/verl/verl/utils/reward_score/simpleqa.py}
RUN_EVAL=${RUN_EVAL:-1}
SKIP_TRAIN=${SKIP_TRAIN:-0}
USE_JUDGE=${USE_JUDGE:-1}
PRUNE_CHECKPOINTS=${PRUNE_CHECKPOINTS:-1}
PROMPT_LEN=${PROMPT_LEN:-2048}
RESP_LEN=${RESP_LEN:-512}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
ROLLOUT_NAME=${ROLLOUT_NAME:-vllm}
ROLLOUT_N=${ROLLOUT_N:-2}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
SAVE_FREQ=${SAVE_FREQ:-200}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
NNODES=${NNODES:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

CKPT_ROOT="outputs/rl/$PROJECT_NAME/$EXPERIMENT_NAME"
PLOT_DIR=${PLOT_DIR:-"$CKPT_ROOT/plots"}

spin_train() {
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

	CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python3 -m recipe.spin.main_spin \
		data.train_files="$TRAIN_DATA" \
		data.val_files="$VAL_DATA" \
		data.train_batch_size="$TRAIN_BATCH_SIZE" \
		data.prompt_key=prompt \
		data.reward_fn_key=data_source \
		actor_rollout_ref.model.path="$MODEL_PATH" \
		actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
		actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
		actor_rollout_ref.rollout.name="$ROLLOUT_NAME" \
		actor_rollout_ref.rollout.n="$ROLLOUT_N" \
		actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
		actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
		custom_reward_function.path="$REWARD_FN_PATH" \
		custom_reward_function.name=compute_score \
		trainer.project_name="$PROJECT_NAME" \
		trainer.experiment_name="$EXPERIMENT_NAME" \
		trainer.default_local_dir="$CKPT_ROOT" \
		trainer.resume_mode=auto \
		trainer.ref_update_freq=-1 \
		trainer.total_epochs="$TOTAL_EPOCHS" \
		trainer.save_freq="$SAVE_FREQ" \
		trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
		trainer.nnodes="$NNODES"
}

spin_eval() {
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

spin_prune() {
	if [[ "$PRUNE_CHECKPOINTS" != "1" ]]; then
		return 0
	fi
	echo "Pruning checkpoints in $CKPT_ROOT"
	python3 "$VERL_DIR/scripts/prune_experiment_checkpoints.py" "$CKPT_ROOT"
}

spin_run() {
	spin_train
	spin_eval
	spin_prune
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
	spin_run
fi
