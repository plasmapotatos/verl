#!/usr/bin/env bash
set -e
set -x

export HYDRA_FULL_ERROR=1

PROJECT_NAME=${PROJECT_NAME:-verl_examples}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-simpleqa_spin}
TRAIN_DATA=${TRAIN_DATA:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/rich_sft/simpleqa_rich_sft_train_origqa.parquet}
VAL_DATA=${VAL_DATA:-$TRAIN_DATA}
EVAL_DATA=${EVAL_DATA:-/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/augment/rich_sft/simpleqa_rich_sft_train_frac0.1_origqa.parquet}
MODEL_PATH=${MODEL_PATH:-/work/hdd/bbsg/twei2/rl/verl/outputs/simpleqa_rich_sft/sft_lr1.5e-4_ep10_seed1/global_step_370/merged_hf_model}
REWARD_FN_PATH=${REWARD_FN_PATH:-/work/hdd/bbsg/twei2/rl/verl/experiments/rl/simpleqa_reward.py}
RUN_EVAL=${RUN_EVAL:-1}
SKIP_TRAIN=${SKIP_TRAIN:-0}
USE_JUDGE=${USE_JUDGE:-1}
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
	STEP0_DIR="$CKPT_ROOT/global_step_0"
	mkdir -p "$STEP0_DIR/generations"

	CKPT_DIRS=$(find "$CKPT_ROOT" -maxdepth 1 -type d -name "global_step_*" | sort -V)
	if [[ -z "$CKPT_DIRS" ]]; then
		CKPT_DIRS="$STEP0_DIR"
	fi

	for CKPT_DIR in $CKPT_DIRS; do
		STEP_NAME=$(basename "$CKPT_DIR")
		if [[ "$STEP_NAME" == "global_step_0" ]]; then
			MERGED_DIR="$MODEL_PATH"
		else
			MERGED_DIR="$CKPT_DIR/merged_hf_model"
			if [[ ! -d "$MERGED_DIR" ]]; then
				python3 -m verl.model_merger merge \
					--backend fsdp \
					--local_dir "$CKPT_DIR/actor" \
					--target_dir "$MERGED_DIR"
			fi
		fi

		GEN_DIR="$CKPT_DIR/generations"
		mkdir -p "$GEN_DIR"

		for EVAL_DATA_ITEM in $EVAL_DATA; do
			EVAL_TAG="$(basename "${EVAL_DATA_ITEM%.parquet}")"
			GEN_OUT="$GEN_DIR/${EXPERIMENT_NAME}_${STEP_NAME}__on_${EVAL_TAG}.parquet"
			EVAL_OUT="${GEN_OUT%.parquet}_eval.json"

			if [[ -f "$EVAL_OUT" ]]; then
				echo "Evaluation already exists for $STEP_NAME on $EVAL_TAG, skipping."
				continue
			fi

			python3 -m verl.trainer.main_generation \
				trainer.nnodes=1 \
				trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
				data.path="$EVAL_DATA_ITEM" \
				data.prompt_key=prompt \
				data.n_samples=1 \
				data.output_path="$GEN_OUT" \
				model.path="$MERGED_DIR" \
				+model.trust_remote_code=True \
				rollout.temperature=0 \
				rollout.prompt_length="$PROMPT_LEN" \
				rollout.response_length="$RESP_LEN" \
				rollout.tensor_model_parallel_size=1 \
				rollout.gpu_memory_utilization=0.8

			if [[ "$USE_JUDGE" == "1" ]]; then
				python3 -m verl.eval.cli \
					--dataset simpleqa \
					--input "$GEN_OUT" \
					--output "$EVAL_OUT" \
					--use-judge
			else
				python3 -m verl.eval.cli \
					--dataset simpleqa \
					--input "$GEN_OUT" \
					--output "$EVAL_OUT"
			fi
		done
	done

	python3 scripts/plot_sft_eval_metrics.py \
		--project-dir "$CKPT_ROOT" \
		--output-dir "$PLOT_DIR"
}

spin_run() {
	spin_train
	spin_eval
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
	spin_run
fi
