#!/usr/bin/env bash
set -euo pipefail

# RL-only chain: skips SFT (uses an existing checkpoint) and submits the RL job.

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
RL_SCRIPT="$WORKDIR/experiments/rich_qa_paraphrased_smoketest_v2/rl/rich_qa_paraphrased_smoketest_v2_rl.sh"

exec bash "${WORKDIR}/chain_job.sh" \
    --gpus 4 \
    --time 02:00:00 \
    "${RL_SCRIPT}"
