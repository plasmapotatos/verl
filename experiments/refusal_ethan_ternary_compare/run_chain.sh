#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
EXP_DIR="$WORKDIR/experiments/refusal_ethan_ternary_compare"
SFT_SCRIPT="$EXP_DIR/sft/refusal_ethan_ternary_compare_sft.sh"

RL_SCRIPTS=(
    "$EXP_DIR/rl/binary_rl.sh"
    "$EXP_DIR/rl/ternary_static_p0.1_rl.sh"
    "$EXP_DIR/rl/ternary_static_p0.5_rl.sh"
    "$EXP_DIR/rl/ternary_static_p1.0_rl.sh"
    "$EXP_DIR/rl/ternary_adaptive_p0.1_rl.sh"
    "$EXP_DIR/rl/ternary_adaptive_p0.5_rl.sh"
    "$EXP_DIR/rl/ternary_adaptive_p1.0_rl.sh"
)

# After SFT completes, launch all RL jobs in parallel via chain_job.
ON_COMPLETE=""
for rl in "${RL_SCRIPTS[@]}"; do
    ON_COMPLETE+="bash ${WORKDIR}/chain_job.sh --gpus 4 --time 02:00:00 ${rl} & "
done
ON_COMPLETE+="wait"

exec bash "${WORKDIR}/chain_job.sh" \
    --gpus 4 \
    --time 02:00:00 \
    --on-complete "${ON_COMPLETE}" \
    "${SFT_SCRIPT}"
