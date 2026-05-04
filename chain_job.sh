#!/bin/bash
# Generic SLURM chain driver: resubmits a target script until it writes
# ${CHAIN_FLAG_FILE} ("complete.flag") or MAX_ITERS is reached.
#
# All project-specific bits (SBATCH account/partition, container exec prefix,
# env setup) live in a separate setup script sourced via --setup. By default
# chain_job.sh looks for chain_setup.sh alongside itself.
#
# See chain_setup.sh for the contract the setup script must satisfy.

set -uo pipefail

trap 'echo "[chain_job] received SIGTERM (time limit or scancel) — will fall through after srun"' TERM
trap 'echo "[chain_job] received SIGINT"' INT

SELF="$(readlink -f "$0")"
SELF_DIR="$(dirname "$SELF")"

DEFAULT_GPUS=4
DEFAULT_TIME="02:00:00"
DEFAULT_MAX_ITERS=20
DEFAULT_SETUP="${SELF_DIR}/chain_setup.sh"

usage() {
    cat >&2 <<EOF
Usage: $0 [--setup PATH] [--gpus N] [--time HH:MM:SS] [--on-complete CMD]
          [--max-iters N] [--complete-on-success] <target_script.sh>

Options:
  --setup PATH       Project-specific setup script (default: ${DEFAULT_SETUP}).
                     Must define CHAIN_WORKDIR, CHAIN_SBATCH_ARGS, and the
                     shell function chain_exec_target().
  --gpus N           GPUs per node (default: $DEFAULT_GPUS).
  --time HH:MM:SS    Wall-clock time limit per iteration (default: $DEFAULT_TIME).
  --on-complete CMD  Shell command to run on the host when the chain completes.
  --max-iters N      Max iterations before giving up (default: $DEFAULT_MAX_ITERS).
    --complete-on-success
                                         (default ON) If target exits rc=0, chain_job.sh creates
                                         complete.flag automatically (target script does not need
                                         to touch it).
    --no-complete-on-success
                                         Disable the above; require the target script to touch
                                         \$CHAIN_FLAG_FILE itself to end the chain.
EOF
    exit 1
}

GPUS="$DEFAULT_GPUS"
TIME_LIMIT="$DEFAULT_TIME"
MAX_ITERS="$DEFAULT_MAX_ITERS"
ON_COMPLETE=""
COMPLETE_ON_SUCCESS=1
SETUP_SCRIPT="$DEFAULT_SETUP"
TARGET_SCRIPT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --setup) [[ $# -ge 2 ]] || usage; SETUP_SCRIPT="$2"; shift 2 ;;
        --setup=*) SETUP_SCRIPT="${1#--setup=}"; shift ;;
        --gpus) [[ $# -ge 2 ]] || usage; GPUS="$2"; shift 2 ;;
        --gpus=*) GPUS="${1#--gpus=}"; shift ;;
        --time) [[ $# -ge 2 ]] || usage; TIME_LIMIT="$2"; shift 2 ;;
        --time=*) TIME_LIMIT="${1#--time=}"; shift ;;
        --on-complete) [[ $# -ge 2 ]] || usage; ON_COMPLETE="$2"; shift 2 ;;
        --on-complete=*) ON_COMPLETE="${1#--on-complete=}"; shift ;;
        --max-iters) [[ $# -ge 2 ]] || usage; MAX_ITERS="$2"; shift 2 ;;
        --max-iters=*) MAX_ITERS="${1#--max-iters=}"; shift ;;
        --complete-on-success) COMPLETE_ON_SUCCESS=1; shift ;;
        --no-complete-on-success) COMPLETE_ON_SUCCESS=0; shift ;;
        -h|--help) usage ;;
        -*) echo "Unknown option: $1" >&2; usage ;;
        *)
            [[ -z "$TARGET_SCRIPT" ]] || usage
            TARGET_SCRIPT="$1"
            shift
            ;;
    esac
done

[[ -n "$TARGET_SCRIPT" ]] || usage
[[ "$GPUS" =~ ^[0-9]+$ ]] || { echo "ERROR: --gpus must be an integer, got '$GPUS'" >&2; exit 1; }
[[ "$MAX_ITERS" =~ ^[0-9]+$ ]] || { echo "ERROR: --max-iters must be an integer, got '$MAX_ITERS'" >&2; exit 1; }

if [[ ! -f "$SETUP_SCRIPT" ]]; then
    echo "ERROR: setup script not found: $SETUP_SCRIPT" >&2
    echo "Pass --setup PATH or drop a chain_setup.sh next to chain_job.sh." >&2
    exit 1
fi
if [[ ! -f "$TARGET_SCRIPT" ]]; then
    echo "ERROR: target script not found: $TARGET_SCRIPT" >&2
    exit 1
fi

SETUP_SCRIPT="$(readlink -f "$SETUP_SCRIPT")"
TARGET_SCRIPT="$(readlink -f "$TARGET_SCRIPT")"

# Setup-script-provided config (with defaults)
CHAIN_WORKDIR="$PWD"
CHAIN_ROOT=""
LOG_ROOT=""
CHAIN_SBATCH_ARGS=()

# shellcheck disable=SC1090
source "$SETUP_SCRIPT"

: "${CHAIN_ROOT:=${CHAIN_WORKDIR}/chain_state}"
: "${LOG_ROOT:=${CHAIN_WORKDIR}/logs/chain}"

if ! declare -f chain_exec_target >/dev/null; then
    echo "ERROR: setup script ${SETUP_SCRIPT} must define chain_exec_target()" >&2
    exit 1
fi

RUN_ID="$(basename "$TARGET_SCRIPT" .sh)"
CHAIN_DIR="${CHAIN_ROOT}/${RUN_ID}"
FLAG_FILE="${CHAIN_DIR}/complete.flag"
STOP_FILE="${CHAIN_DIR}/stop.flag"
COUNTER_FILE="${CHAIN_DIR}/counter"
LOG_DIR="${LOG_ROOT}/${RUN_ID}"

mkdir -p "$CHAIN_DIR" "$LOG_DIR"
cd "$CHAIN_WORKDIR"

submit_next() {
    sbatch \
        "${CHAIN_SBATCH_ARGS[@]}" \
        --job-name="chain_${RUN_ID}" \
        --gpus-per-node="${GPUS}" \
        --time="${TIME_LIMIT}" \
        --signal=B:TERM@120 \
        --output="${LOG_DIR}/iter_%j.out" \
        --error="${LOG_DIR}/iter_%j.err" \
        "$SELF" \
            --setup "$SETUP_SCRIPT" \
            --gpus "$GPUS" \
            --time "$TIME_LIMIT" \
            --max-iters "$MAX_ITERS" \
            $([[ "$COMPLETE_ON_SUCCESS" == "1" ]] && echo "--complete-on-success" || echo "--no-complete-on-success") \
            "$TARGET_SCRIPT"
}

# If not running under SLURM, this is the initial launch: sbatch ourselves and exit.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    EXISTING="$(squeue -h -u "$USER" --name="chain_${RUN_ID}" -o '%i %T' 2>/dev/null || true)"
    if [[ -n "$EXISTING" ]]; then
        echo "[chain_job] WARNING: a chain with RUN_ID='${RUN_ID}' is already in the queue:" >&2
        echo "$EXISTING" | sed 's/^/    /' >&2
        echo "" >&2
        echo "Launching another one will cause both chains to race on:" >&2
        echo "    ${CHAIN_DIR}/{counter,complete.flag,stop.flag}" >&2
        echo "    ${LOG_DIR}/" >&2
        echo "" >&2
        echo "If you really want to proceed, first kill the existing job(s) with:" >&2
        echo "    ${CHAIN_WORKDIR}/nuke.sh $(echo "$EXISTING" | awk '{print $1}' | tr '\n' ' ')" >&2
        echo "Or re-run with FORCE=1 ./chain_job.sh ... to override this check." >&2
        if [[ "${FORCE:-0}" != "1" ]]; then
            exit 1
        fi
        echo "[chain_job] FORCE=1 set — proceeding anyway." >&2
    fi

    echo 0 > "$COUNTER_FILE"
    rm -f "$FLAG_FILE" "$STOP_FILE"
    if [[ -n "$ON_COMPLETE" ]]; then
        echo "$ON_COMPLETE" > "${CHAIN_DIR}/on_complete.sh"
    else
        rm -f "${CHAIN_DIR}/on_complete.sh"
    fi
    echo "[chain_job] Initial launch for RUN_ID=${RUN_ID}"
    echo "[chain_job] Setup:     ${SETUP_SCRIPT}"
    echo "[chain_job] State dir: ${CHAIN_DIR}"
    echo "[chain_job] Log dir:   ${LOG_DIR}"
    echo "[chain_job] GPUs per node: ${GPUS}"
    echo "[chain_job] Time limit: ${TIME_LIMIT}"
    echo "[chain_job] Complete-on-success: ${COMPLETE_ON_SUCCESS}"
    if [[ -n "$ON_COMPLETE" ]]; then
        echo "[chain_job] On-complete: ${ON_COMPLETE}"
    fi
    echo "[chain_job] Submitting first batch job..."
    submit_next
    exit 0
fi

# --- Running under SLURM from here on ---

ITER=$(<"$COUNTER_FILE")
ITER=$((ITER + 1))
echo "$ITER" > "$COUNTER_FILE"

# Compute SLURM job end time on the host (container usually lacks squeue)
# so verl's should_save_ckpt_esi() can force a checkpoint near wall-clock limit.
JOB_END_TS=""
JOB_END_HUMAN="$(squeue -h -j "$SLURM_JOB_ID" -o '%e' 2>/dev/null | tr -d '[:space:]')"
if [[ -n "$JOB_END_HUMAN" && "$JOB_END_HUMAN" != "N/A" ]]; then
    JOB_END_TS="$(date -d "$JOB_END_HUMAN" +%s 2>/dev/null || true)"
fi

echo "==========================================================="
echo "[chain_job] RUN_ID=${RUN_ID}"
echo "[chain_job] Iteration: ${ITER} / ${MAX_ITERS}"
echo "[chain_job] SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "[chain_job] GPUs per node: ${GPUS}"
echo "[chain_job] Setup:  ${SETUP_SCRIPT}"
echo "[chain_job] Target: ${TARGET_SCRIPT}"
echo "[chain_job] Chain dir: ${CHAIN_DIR}"
echo "[chain_job] Log dir:   ${LOG_DIR}"
if [[ -n "$JOB_END_TS" ]]; then
    echo "[chain_job] Job end: ${JOB_END_HUMAN} (${JOB_END_TS})"
else
    echo "[chain_job] WARN: could not resolve job end time."
fi
echo "==========================================================="

# Exported so chain_exec_target (and the target script) can see them
export RUN_ID CHAIN_DIR JOB_END_TS
export CHAIN_FLAG_FILE="$FLAG_FILE"
# Propagate GPU count so target scripts can size NPROC/batching accordingly.
export N_GPUS_PER_NODE="$GPUS"

# Feed the SLURM wall-clock end time into verl's ESI hook so it force-saves
# (and exits cleanly, post-patch) before scancel. esi_redundant_time in
# grpo.sh is driven by TIME_BUDGET_BUFFER_SEC; default 600s here.
if [[ -n "$JOB_END_TS" ]]; then
    export MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP="$JOB_END_TS"
    export TIME_BUDGET_BUFFER_SEC="${TIME_BUDGET_BUFFER_SEC:-600}"
    echo "[chain_job] Exported MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP=${JOB_END_TS}, TIME_BUDGET_BUFFER_SEC=${TIME_BUDGET_BUFFER_SEC}"
fi

set +e
chain_exec_target "$TARGET_SCRIPT"
TARGET_RC=$?
set -e

echo "[chain_job] Target script exited with rc=${TARGET_RC}"

if [[ "$COMPLETE_ON_SUCCESS" == "1" && "$TARGET_RC" -eq 0 ]]; then
    touch "$FLAG_FILE"
    echo "[chain_job] complete-on-success enabled; wrote ${FLAG_FILE}."
fi

if [[ -f "$FLAG_FILE" ]]; then
    echo "[chain_job] complete.flag found — chain DONE for ${RUN_ID}."
    COMPLETED_LOG="${LOG_ROOT}/completed_experiments.tsv"
    if [[ ! -f "$COMPLETED_LOG" ]]; then
        printf 'finished_at\trun_id\titerations\tfinal_job_id\ttarget_script\tchain_dir\n' > "$COMPLETED_LOG"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" \
        "$RUN_ID" \
        "$ITER" \
        "$SLURM_JOB_ID" \
        "$TARGET_SCRIPT" \
        "$CHAIN_DIR" \
        >> "$COMPLETED_LOG"
    echo "[chain_job] Logged completion to ${COMPLETED_LOG}"
    ON_COMPLETE_FILE="${CHAIN_DIR}/on_complete.sh"
    if [[ -f "$ON_COMPLETE_FILE" ]]; then
        ON_COMPLETE_CMD="$(<"$ON_COMPLETE_FILE")"
        echo "[chain_job] Running on-complete hook: ${ON_COMPLETE_CMD}"
        ( cd "$CHAIN_WORKDIR" && unset SLURM_JOB_ID && eval "$ON_COMPLETE_CMD" )
        ON_COMPLETE_RC=$?
        echo "[chain_job] on-complete hook exited with rc=${ON_COMPLETE_RC}"
    fi
    exit 0
fi

# User-requested stop (set this BEFORE running scancel to prevent resubmission)
if [[ -f "$STOP_FILE" ]]; then
    echo "[chain_job] stop.flag found — chain STOPPED by user for ${RUN_ID}."
    exit 0
fi

if [[ "$ITER" -ge "$MAX_ITERS" ]]; then
    echo "[chain_job] Reached MAX_ITERS=${MAX_ITERS}; not resubmitting." >&2
    exit 1
fi

echo "[chain_job] No complete.flag — resubmitting (next iter $((ITER + 1)))."
submit_next
