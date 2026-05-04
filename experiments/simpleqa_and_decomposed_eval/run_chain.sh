#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
RL_SCRIPT="$WORKDIR/experiments/simpleqa_and_decomposed_eval/rl/simpleqa_and_decomposed_eval_rl.sh"

exec bash "${WORKDIR}/chain_job.sh" \
    --gpus 4 \
    --time 02:00:00 \
    "${RL_SCRIPT}"
