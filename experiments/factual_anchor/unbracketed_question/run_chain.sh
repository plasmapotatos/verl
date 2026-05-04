#!/usr/bin/env bash
# Chain the unbracketed_question SFT run, and on completion chain the matching
# RL run that trains on top of the last SFT checkpoint with the
# correct + 0.5 * novel_bag reward.
set -euo pipefail

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
SFT_SCRIPT="$WORKDIR/experiments/factual_anchor/unbracketed_question/sft/train_sft.sh"
RL_SCRIPT="$WORKDIR/experiments/factual_anchor/unbracketed_question/rl/train_rl.sh"

# The on-complete hook is executed on the SLURM host (outside the container)
# after the SFT chain writes its complete.flag.  It kicks off another chain
# targeting the RL script, inheriting the same GPU/time settings used for SFT.
ON_COMPLETE="bash ${WORKDIR}/chain_job.sh --gpus 2 --time 02:00:00 ${RL_SCRIPT}"

exec bash "${WORKDIR}/chain_job.sh" \
    --gpus 1 \
    --time 02:00:00 \
    --on-complete "${ON_COMPLETE}" \
    "${SFT_SCRIPT}"
