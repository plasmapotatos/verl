#!/usr/bin/env bash
set -e

# Evaluate the RL step_1280 checkpoint on the k=3 held-out eval split(s).
# Writes generations + judge eval into the checkpoint's existing generations/ dir.

CKPT_DIR="/work/hdd/bbsg/twei2/rl/verl/outputs/rl/simpleqa_decompose_mixed_k12_smoke_grpo/binary_decompose_mixed_k12_smoke/global_step_1280"
EXPERIMENT_NAME="binary_decompose_mixed_k12_smoke"
STEP_NAME="global_step_1280"
GEN_PREFIX="${EXPERIMENT_NAME}_${STEP_NAME}"
MODEL_DIR="$CKPT_DIR/merged_hf_model"
GEN_DIR="$CKPT_DIR/generations"

DATASET="${DATASET:-simpleqa}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
RESP_LEN="${RESP_LEN:-512}"   # k=3 answers are longer than k=2

EVAL_DATA_LIST=(
  "/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose_mixed_k12_smoke/sft/train_eval_k3.parquet"
  "/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose_mixed_k12_smoke/sft/train_eval_k3_cot.parquet"
)

mkdir -p "$GEN_DIR"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "merged_hf_model missing at $MODEL_DIR — running merge"
  bash /work/hdd/bbsg/twei2/rl/verl/experiments/utils/merge_checkpoint.sh "$CKPT_DIR"
fi

for EVAL_PATH in "${EVAL_DATA_LIST[@]}"; do
  EVAL_TAG="$(basename "${EVAL_PATH%.parquet}")"
  GEN_OUT="$GEN_DIR/${GEN_PREFIX}__on_${EVAL_TAG}.parquet"
  EVAL_OUT="${GEN_OUT%.parquet}_eval.json"

  echo "[$STEP_NAME] eval_data=$(basename "$EVAL_PATH")"

  if [[ -f "$GEN_OUT" && -s "$GEN_OUT" ]]; then
    echo "  generation exists: $(basename "$GEN_OUT")"
  else
    echo "  generating -> $GEN_OUT"
    python3 -m verl.trainer.main_generation \
      trainer.nnodes=1 \
      trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
      data.path="$EVAL_PATH" \
      data.prompt_key=prompt \
      data.n_samples=1 \
      data.output_path="$GEN_OUT" \
      model.path="$MODEL_DIR" \
      +model.trust_remote_code=True \
      rollout.temperature=0 \
      rollout.prompt_length="$PROMPT_LEN" \
      rollout.response_length="$RESP_LEN" \
      rollout.tensor_model_parallel_size=1 \
      rollout.gpu_memory_utilization=0.8
  fi

  if [[ -f "$EVAL_OUT" ]]; then
    echo "  eval exists: $(basename "$EVAL_OUT")"
  else
    python3 -m verl.eval.cli --dataset "$DATASET" --input "$GEN_OUT" --output "$EVAL_OUT" --use-judge
  fi
done

echo "Done. Outputs in $GEN_DIR"
