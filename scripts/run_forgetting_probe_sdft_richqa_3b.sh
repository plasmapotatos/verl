#!/usr/bin/env bash
set -euo pipefail

# Catastrophic forgetting probe: run scripts/run_infer_and_eval.sh on
# knows.parquet for both the SDFT teacher-gen and student checkpoints
# at step 1000.

VERL_DIR="/work/hdd/bbsg/twei2/rl/verl"
PROBE_DATA="$VERL_DIR/data/simpleqa/partition/knows.parquet"

CKPTS=(
    "/work/hdd/bbsg/twei2/rl/Self-Distillation/outputs/sdft_richqa_smoke_3b_teachergen/checkpoint-1000"
    "/work/hdd/bbsg/twei2/rl/Self-Distillation/outputs/sdft_richqa_smoke_3b_student/checkpoint-1000"
)

export NGPU_GEN="${NGPU_GEN:-4}"
export N_SAMPLES="${N_SAMPLES:-1}"
export TEMP="${TEMP:-0}"
export PROMPT_LEN="${PROMPT_LEN:-2048}"
export RESP_LEN="${RESP_LEN:-1024}"
export TP_SIZE="${TP_SIZE:-1}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
export USE_JUDGE="${USE_JUDGE:-1}"
export DATASET="${DATASET:-simpleqa}"

cd "$VERL_DIR"

for CKPT in "${CKPTS[@]}"; do
    OUT_DIR="$CKPT/forgetting_probe_knows"
    mkdir -p "$OUT_DIR"
    echo "=========================================================="
    echo "[probe] ckpt: $CKPT"
    echo "[probe] data: $PROBE_DATA"
    echo "[probe] out:  $OUT_DIR"
    echo "=========================================================="
    bash "$VERL_DIR/scripts/run_infer_and_eval.sh" \
        "$PROBE_DATA" \
        "$CKPT" \
        "$OUT_DIR"
done

echo ""
echo "DONE. Eval JSONs:"
for CKPT in "${CKPTS[@]}"; do
    ls -1 "$CKPT/forgetting_probe_knows"/*_eval.json 2>/dev/null || true
done
