#!/usr/bin/env bash
# Chain the decompose_and_richqa SFT run, and on completion chain the matching
# GRPO run that trains on top of the last SFT checkpoint.
set -euo pipefail

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
SFT_SCRIPT="$WORKDIR/experiments/decompose_and_richqa/sft/decompose_and_richqa_sft.sh"
RL_SCRIPT="$WORKDIR/experiments/decompose_and_richqa/rl/decompose_and_richqa_rl.sh"

# The on-complete hook is executed on the SLURM host (outside the container)
# after the SFT chain writes its complete.flag.  It kicks off another chain
# targeting the RL script.
ON_COMPLETE="bash ${WORKDIR}/chain_job.sh --gpus 4 --time 02:00:00 ${RL_SCRIPT}"

exec bash "${WORKDIR}/chain_job.sh" \
    --gpus 4 \
    --time 02:00:00 \
    --on-complete "${ON_COMPLETE}" \
    "${SFT_SCRIPT}"
