#!/bin/bash
# Cleanly kill a chain_job.sh chain by touching its stop.flag, then scancel.
# Usage: ./nuke.sh <job_id> [<job_id> ...]

set -uo pipefail

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
CHAIN_ROOT="${WORKDIR}/chain_state"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <job_id> [<job_id> ...]" >&2
    echo "       Looks up each job's name (expects 'chain_<RUN_ID>'), touches" >&2
    echo "       \${CHAIN_ROOT}/<RUN_ID>/stop.flag, then scancels the job." >&2
    exit 1
fi

for JOBID in "$@"; do
    echo "=== nuke job ${JOBID} ==="

    # Get the job name from squeue (works for pending/running). Fall back to sacct.
    JOBNAME="$(squeue -j "$JOBID" -h -o '%j' 2>/dev/null | head -n1 | tr -d '[:space:]')"
    if [[ -z "$JOBNAME" ]]; then
        JOBNAME="$(sacct -j "$JOBID" -n -P -o JobName 2>/dev/null | head -n1 | tr -d '[:space:]')"
    fi

    if [[ -z "$JOBNAME" ]]; then
        echo "  WARN: could not find job name for ${JOBID} — scancelling anyway, but you may need to touch stop.flag yourself." >&2
        scancel "$JOBID"
        continue
    fi

    echo "  job name: ${JOBNAME}"

    if [[ "$JOBNAME" != chain_* ]]; then
        echo "  WARN: job ${JOBID} (${JOBNAME}) is not a chain_* job — scancelling without stop.flag." >&2
        scancel "$JOBID"
        continue
    fi

    RUN_ID="${JOBNAME#chain_}"
    CHAIN_DIR="${CHAIN_ROOT}/${RUN_ID}"
    STOP_FILE="${CHAIN_DIR}/stop.flag"

    if [[ ! -d "$CHAIN_DIR" ]]; then
        echo "  WARN: chain dir ${CHAIN_DIR} does not exist — creating it so the stop.flag exists." >&2
        mkdir -p "$CHAIN_DIR"
    fi

    touch "$STOP_FILE"
    echo "  stop.flag: ${STOP_FILE}"

    scancel "$JOBID"
    echo "  scancelled ${JOBID}"
done

echo
echo "Done. Verify with:  squeue -u \$USER"
