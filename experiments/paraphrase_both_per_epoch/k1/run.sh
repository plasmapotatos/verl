#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
SFT_SCRIPT="$WORKDIR/experiments/paraphrase_both_per_epoch/k1/k1_sft.sh"

exec bash "${WORKDIR}/chain_job.sh" \
    --gpus 4 \
    --time 02:00:00 \
    "${SFT_SCRIPT}"
