#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
RL_SCRIPT="$WORKDIR/experiments/decompose_mixed_k12_smoke_rl_k3/rl/decompose_mixed_k12_smoke_rl_k3_rl.sh"

exec bash "${WORKDIR}/chain_job.sh" \
    --gpus 4 \
    --time 02:00:00 \
    "${RL_SCRIPT}"
