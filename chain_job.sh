#!/bin/bash
#SBATCH --account=bbsg-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00

set -uo pipefail

# Don't let SIGTERM from SLURM (time limit / scancel) kill the wrapper bash before
# we can reach the resubmission logic. We just want srun to die; bash itself must
# survive past the srun call so we can sbatch the next iteration.
trap 'echo "[chain_job] received SIGTERM (time limit or scancel) — will fall through after srun"' TERM
trap 'echo "[chain_job] received SIGINT"' INT

# $VERL is the container exec prefix used by other slurm scripts in this repo
# (e.g. experiments/jobs/batch_25_75.slurm). Default it if not set in env.
: "${VERL:=apptainer exec --nv --bind /work,/u /work/hdd/bbsg/twei2/rl/torch2501.sif}"
export VERL

MAX_ITERS=20
WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
CHAIN_ROOT="${WORKDIR}/chain_state"
LOG_ROOT="${WORKDIR}/logs/chain"
DEFAULT_GPUS=2

DEFAULT_TIME="02:00:00"

usage() {
    echo "Usage: $0 [--gpus N] [--time HH:MM:SS] [--on-complete CMD] <target_script.sh>" >&2
    echo "  --gpus N           GPUs per node (default: $DEFAULT_GPUS)" >&2
    echo "  --time HH:MM:SS   Wall-clock time limit per iteration (default: $DEFAULT_TIME)" >&2
    echo "  --on-complete CMD  Shell command to run on the host when the chain completes." >&2
    echo "                     Runs after complete.flag is found. Useful for launching" >&2
    echo "                     downstream jobs (e.g. RL after SFT)." >&2
    exit 1
}

GPUS="$DEFAULT_GPUS"
TIME_LIMIT="$DEFAULT_TIME"
ON_COMPLETE=""
TARGET_SCRIPT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            [[ $# -ge 2 ]] || usage
            GPUS="$2"
            shift 2
            ;;
        --gpus=*)
            GPUS="${1#--gpus=}"
            shift
            ;;
        --time)
            [[ $# -ge 2 ]] || usage
            TIME_LIMIT="$2"
            shift 2
            ;;
        --time=*)
            TIME_LIMIT="${1#--time=}"
            shift
            ;;
        --on-complete)
            [[ $# -ge 2 ]] || usage
            ON_COMPLETE="$2"
            shift 2
            ;;
        --on-complete=*)
            ON_COMPLETE="${1#--on-complete=}"
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            ;;
        *)
            [[ -z "$TARGET_SCRIPT" ]] || usage
            TARGET_SCRIPT="$1"
            shift
            ;;
    esac
done

[[ -n "$TARGET_SCRIPT" ]] || usage
[[ "$GPUS" =~ ^[0-9]+$ ]] || { echo "ERROR: --gpus must be an integer, got '$GPUS'" >&2; exit 1; }

if [[ ! -f "$TARGET_SCRIPT" ]]; then
    echo "ERROR: target script not found: $TARGET_SCRIPT" >&2
    exit 1
fi

# Resolve to absolute path so resubmissions work regardless of cwd
TARGET_SCRIPT="$(readlink -f "$TARGET_SCRIPT")"

RUN_ID="$(basename "$TARGET_SCRIPT" .sh)"
CHAIN_DIR="${CHAIN_ROOT}/${RUN_ID}"
FLAG_FILE="${CHAIN_DIR}/complete.flag"
STOP_FILE="${CHAIN_DIR}/stop.flag"
COUNTER_FILE="${CHAIN_DIR}/counter"
LOG_DIR="${LOG_ROOT}/${RUN_ID}"

mkdir -p "$CHAIN_DIR" "$LOG_DIR"
cd "$WORKDIR"

# If not running under SLURM, this is the initial launch: sbatch ourselves and exit.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    # Check for an existing chain with the same RUN_ID to avoid duplicate chains
    # writing to the same chain_state / log dir (they would race on counter,
    # complete.flag, stop.flag, etc).
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
        echo "    ${WORKDIR}/nuke.sh $(echo "$EXISTING" | awk '{print $1}' | tr '\n' ' ')" >&2
        echo "Or re-run with FORCE=1 ./chain_job.sh ... to override this check." >&2
        if [[ "${FORCE:-0}" != "1" ]]; then
            exit 1
        fi
        echo "[chain_job] FORCE=1 set — proceeding anyway." >&2
    fi

    # Fresh launch from the CLI: reset counter and clear any stale flags.
    echo 0 > "$COUNTER_FILE"
    rm -f "$FLAG_FILE" "$STOP_FILE"
    # Persist on-complete hook so chained iterations can read it
    if [[ -n "$ON_COMPLETE" ]]; then
        echo "$ON_COMPLETE" > "${CHAIN_DIR}/on_complete.sh"
    else
        rm -f "${CHAIN_DIR}/on_complete.sh"
    fi
    echo "[chain_job] Initial launch for RUN_ID=${RUN_ID}"
    echo "[chain_job] State dir: ${CHAIN_DIR}"
    echo "[chain_job] Log dir:   ${LOG_DIR}"
    echo "[chain_job] GPUs per node: ${GPUS}"
    echo "[chain_job] Time limit: ${TIME_LIMIT}"
    if [[ -n "$ON_COMPLETE" ]]; then
        echo "[chain_job] On-complete: ${ON_COMPLETE}"
    fi
    echo "[chain_job] Submitting first batch job..."
    sbatch \
        --job-name="chain_${RUN_ID}" \
        --gpus-per-node="${GPUS}" \
        --time="${TIME_LIMIT}" \
        --signal=B:TERM@120 \
        --output="${LOG_DIR}/iter_%j.out" \
        --error="${LOG_DIR}/iter_%j.err" \
        "${WORKDIR}/chain_job.sh" --gpus "${GPUS}" --time "${TIME_LIMIT}" "$TARGET_SCRIPT"
    exit 0
fi

# --- Running under SLURM from here on ---

# Increment iteration counter
ITER=$(<"$COUNTER_FILE")
ITER=$((ITER + 1))
echo "$ITER" > "$COUNTER_FILE"

# Compute the SLURM job end time as a unix timestamp so that verl's
# should_save_ckpt_esi() (ray_trainer.py:1361) can force a checkpoint save
# shortly before the wall-clock limit. The container typically doesn't have
# `squeue`, so we resolve this on the host and pass it through as an env var.
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
echo "[chain_job] Target: ${TARGET_SCRIPT}"
echo "[chain_job] Chain dir: ${CHAIN_DIR}"
echo "[chain_job] Log dir:   ${LOG_DIR}"
if [[ -n "$JOB_END_TS" ]]; then
    echo "[chain_job] Job end: ${JOB_END_HUMAN} (${JOB_END_TS}) — passed to verl via MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"
else
    echo "[chain_job] WARN: could not resolve job end time; verl ESI-save will be inactive."
fi
echo "==========================================================="

# Export so target script can write the flag
export RUN_ID
export CHAIN_DIR
export CHAIN_FLAG_FILE="$FLAG_FILE"

# Run the target script inside the container env (don't let its failure kill the chain logic)
set +e
srun $VERL bash -c "source /work/hdd/bbsg/twei2/rl/verl_container_rc.sh && \
                    cd ${WORKDIR} && \
                    RUN_ID='${RUN_ID}' CHAIN_DIR='${CHAIN_DIR}' CHAIN_FLAG_FILE='${FLAG_FILE}' \
                    MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP='${JOB_END_TS}' \
                    bash '${TARGET_SCRIPT}'"
TARGET_RC=$?
set -e

echo "[chain_job] Target script exited with rc=${TARGET_RC}"

# Check completion
if [[ -f "$FLAG_FILE" ]]; then
    echo "[chain_job] complete.flag found — chain DONE for ${RUN_ID}."
    # Run on-complete hook if one was registered
    ON_COMPLETE_FILE="${CHAIN_DIR}/on_complete.sh"
    if [[ -f "$ON_COMPLETE_FILE" ]]; then
        ON_COMPLETE_CMD="$(<"$ON_COMPLETE_FILE")"
        echo "[chain_job] Running on-complete hook: ${ON_COMPLETE_CMD}"
        ( cd "$WORKDIR" && unset SLURM_JOB_ID && eval "$ON_COMPLETE_CMD" )
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
sbatch \
    --job-name="chain_${RUN_ID}" \
    --gpus-per-node="${GPUS}" \
    --time="${TIME_LIMIT}" \
    --signal=B:TERM@120 \
    --output="${LOG_DIR}/iter_%j.out" \
    --error="${LOG_DIR}/iter_%j.err" \
    "${WORKDIR}/chain_job.sh" --gpus "${GPUS}" --time "${TIME_LIMIT}" "$TARGET_SCRIPT"
