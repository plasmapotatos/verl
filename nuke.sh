#!/bin/bash
# Cleanly kill a chain_job.sh chain by touching its stop.flag, then scancel.
# Usage: ./nuke.sh <job_id> [<job_id> ...]

set -uo pipefail

DEFAULT_WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
WORKDIR="${NUKE_WORKDIR:-$DEFAULT_WORKDIR}"
CHAIN_ROOT="${NUKE_CHAIN_ROOT:-${WORKDIR}/chain_state}"

usage() {
    cat >&2 <<EOF
Usage: $0 [--workdir PATH | --chain-root PATH] <job_id> [<job_id> ...]
       Looks up each job's name (expects 'chain_<RUN_ID>'), touches
       \${CHAIN_ROOT}/<RUN_ID>/stop.flag, then scancels the job.

Options:
  --workdir PATH     Project root whose chain_state/ holds the stop.flag.
                     Defaults to \$NUKE_WORKDIR or ${DEFAULT_WORKDIR}.
  --chain-root PATH  Override the chain_state dir directly (takes precedence
                     over --workdir). Defaults to \$NUKE_CHAIN_ROOT or
                     \$WORKDIR/chain_state.
EOF
    exit 1
}

JOBIDS=()
CHAIN_ROOT_EXPLICIT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workdir) [[ $# -ge 2 ]] || usage; WORKDIR="$2"; shift 2 ;;
        --workdir=*) WORKDIR="${1#--workdir=}"; shift ;;
        --chain-root) [[ $# -ge 2 ]] || usage; CHAIN_ROOT="$2"; CHAIN_ROOT_EXPLICIT=1; shift 2 ;;
        --chain-root=*) CHAIN_ROOT="${1#--chain-root=}"; CHAIN_ROOT_EXPLICIT=1; shift ;;
        -h|--help) usage ;;
        -*) echo "Unknown option: $1" >&2; usage ;;
        *) JOBIDS+=("$1"); shift ;;
    esac
done

# If --workdir was set but --chain-root wasn't, re-derive CHAIN_ROOT from the
# (possibly-updated) WORKDIR.
if [[ "$CHAIN_ROOT_EXPLICIT" -eq 0 ]]; then
    CHAIN_ROOT="${NUKE_CHAIN_ROOT:-${WORKDIR}/chain_state}"
fi

if [[ ${#JOBIDS[@]} -lt 1 ]]; then
    usage
fi

set -- "${JOBIDS[@]}"

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
