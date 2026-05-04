#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
SFT_SCRIPT="$WORKDIR/experiments/refusal_clean_richqa_50_50/sft/refusal_clean_richqa_50_50_sft.sh"
RL_SCRIPT="$WORKDIR/experiments/refusal_clean_richqa_50_50/rl/refusal_clean_richqa_50_50_rl.sh"

ON_COMPLETE="bash ${WORKDIR}/chain_job.sh --gpus 4 --time 02:00:00 ${RL_SCRIPT}"

exec bash "${WORKDIR}/chain_job.sh" \
    --gpus 4 \
    --time 02:00:00 \
    --on-complete "${ON_COMPLETE}" \
    "${SFT_SCRIPT}"
