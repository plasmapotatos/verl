#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
SFT_SCRIPT="$WORKDIR/experiments/decompose_mixed_k12_smoke/sft/decompose_mixed_k12_smoke_sft.sh"
RL_SCRIPT="$WORKDIR/experiments/decompose_mixed_k12_smoke/rl/decompose_mixed_k12_smoke_rl.sh"

ON_COMPLETE="bash ${WORKDIR}/chain_job.sh --gpus 4 --time 02:00:00 ${RL_SCRIPT}"

exec bash "${WORKDIR}/chain_job.sh" \
    --gpus 4 \
    --time 02:00:00 \
    --on-complete "${ON_COMPLETE}" \
    "${SFT_SCRIPT}"
