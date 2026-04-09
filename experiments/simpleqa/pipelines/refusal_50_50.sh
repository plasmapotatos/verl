#!/usr/bin/env bash
# Full SFT → RL pipeline for refusal 50/50 partition.
#
# Stage 1: SFT (run_refusal_50_50.sh) via chain_job
# Stage 2: On SFT completion, fan out GRPO + RPP chains in parallel
#
# Usage:
#   bash experiments/simpleqa/pipelines/refusal_50_50.sh
#   bash experiments/simpleqa/pipelines/refusal_50_50.sh --gpus 4 --time 02:00:00
set -euo pipefail

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
cd "$WORKDIR"

GPUS="${GPUS:-2}"
TIME="${TIME:-02:00:00}"

# Parse optional overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --gpus=*) GPUS="${1#--gpus=}"; shift ;;
        --time) TIME="$2"; shift 2 ;;
        --time=*) TIME="${1#--time=}"; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

SFT_SCRIPT="experiments/simpleqa/sft/partition/run_refusal_50_50.sh"
GRPO_SCRIPT="experiments/simpleqa/rl/partition/grpo/grpo_refusal_50_50.sh"
RPP_SCRIPT="experiments/simpleqa/rl/partition/rpp/rpp_refusal_50_50.sh"

echo "=== Refusal 50/50 Pipeline ==="
echo "GPUs: $GPUS"
echo "Time: $TIME"
echo "Stage 1: SFT  → $SFT_SCRIPT"
echo "Stage 2: GRPO → $GRPO_SCRIPT"
echo "         RPP  → $RPP_SCRIPT"
echo "==============================="

./chain_job.sh --gpus "$GPUS" --time "$TIME" \
    --on-complete "./chain_job.sh --gpus $GPUS --time $TIME $GRPO_SCRIPT & \
                   ./chain_job.sh --gpus $GPUS --time $TIME $RPP_SCRIPT &" \
    "$SFT_SCRIPT"
