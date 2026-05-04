#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
SFT_SCRIPT="$WORKDIR/experiments/paraphrase_rich_both/sft/paraphrase_rich_both_sft.sh"
RL_SCRIPT="$WORKDIR/experiments/paraphrase_rich_both/rl/paraphrase_rich_both_rl.sh"
RL_SMOKETEST_SCRIPT="$WORKDIR/experiments/paraphrase_rich_both/rl/paraphrase_rich_both_smoketest_rl.sh"

ON_COMPLETE="bash ${WORKDIR}/chain_job.sh --gpus 4 --time 02:00:00 ${RL_SCRIPT}; bash ${WORKDIR}/chain_job.sh --gpus 4 --time 02:00:00 ${RL_SMOKETEST_SCRIPT}"

exec bash "${WORKDIR}/chain_job.sh" \
    --gpus 4 \
    --time 02:00:00 \
    --on-complete "${ON_COMPLETE}" \
    "${SFT_SCRIPT}"
