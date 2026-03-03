#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_infer_and_eval.sh <input_parquet> <model_path> <output_dir>
#
# Optional env:
#   N_SAMPLES (default: 1)
#   TEMP (default: 0)
#   PROMPT_LEN (default: 2048)
#   RESP_LEN (default: 1024)
#   TP_SIZE (default: 1)
#   GPU_MEM_UTIL (default: 0.8)
#   NGPU_GEN (default: 1)
#   USE_JUDGE (default: 0)
#   DATASET (default: simpleqa)

if [[ "$#" -lt 3 ]]; then
  echo "Usage: bash scripts/run_infer_and_eval.sh <input_parquet> <model_path> <output_dir>"
  exit 1
fi

INPUT_PARQUET="$1"
MODEL_PATH="$2"
OUTPUT_DIR="$3"

N_SAMPLES="${N_SAMPLES:-1}"
TEMP="${TEMP:-0}"
PROMPT_LEN="${PROMPT_LEN:-2048}"
RESP_LEN="${RESP_LEN:-1024}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
NGPU_GEN="${NGPU_GEN:-1}"
USE_JUDGE="${USE_JUDGE:-1}"
DATASET="${DATASET:-simpleqa}"

if [[ ! -f "$INPUT_PARQUET" ]]; then
  echo "ERROR: input parquet not found: $INPUT_PARQUET"
  exit 1
fi
if [[ ! -d "$MODEL_PATH" && ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: model path not found: $MODEL_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

base_name="$(basename "$INPUT_PARQUET")"
base_stem="${base_name%.parquet}"
OUT_PARQUET="$OUTPUT_DIR/${base_stem}__gen.parquet"
OUT_JSON="$OUTPUT_DIR/${base_stem}__gen_eval.json"

if [[ -f "$OUT_PARQUET" ]]; then
  echo "Generation already exists: $OUT_PARQUET (skipping)"
else
  echo "Generating -> $OUT_PARQUET"
  python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="$NGPU_GEN" \
    data.path="$INPUT_PARQUET" \
    data.prompt_key=prompt \
    data.n_samples="$N_SAMPLES" \
    data.output_path="$OUT_PARQUET" \
    model.path="$MODEL_PATH" \
    +model.trust_remote_code=True \
    rollout.temperature="$TEMP" \
    rollout.prompt_length="$PROMPT_LEN" \
    rollout.response_length="$RESP_LEN" \
    rollout.tensor_model_parallel_size="$TP_SIZE" \
    rollout.gpu_memory_utilization="$GPU_MEM_UTIL"
fi

if [[ -f "$OUT_JSON" ]]; then
  echo "Evaluation already exists: $OUT_JSON (skipping)"
  exit 0
fi

echo "Evaluating -> $OUT_JSON"
if [[ "$USE_JUDGE" == "1" ]]; then
  python3 -m verl.eval.cli \
    --dataset "$DATASET" \
    --input "$OUT_PARQUET" \
    --output "$OUT_JSON" \
    --use-judge
else
  python3 -m verl.eval.cli \
    --dataset "$DATASET" \
    --input "$OUT_PARQUET" \
    --output "$OUT_JSON"
fi
