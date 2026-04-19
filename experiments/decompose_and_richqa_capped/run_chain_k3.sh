#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
SFT_SCRIPT="$WORKDIR/experiments/decompose_and_richqa_capped/sft/decompose_and_richqa_capped_sft_k3.sh"
RL_SCRIPT="$WORKDIR/experiments/decompose_and_richqa_capped/rl/decompose_and_richqa_capped_rl_k3.sh"

ON_COMPLETE="bash ${WORKDIR}/chain_job.sh --gpus 4 --time 02:00:00 ${RL_SCRIPT}"

exec bash "${WORKDIR}/chain_job.sh" \
    --gpus 4 \
    --time 02:00:00 \
    --on-complete "${ON_COMPLETE}" \
    "${SFT_SCRIPT}"
