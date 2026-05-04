# chain_setup.sh — project-specific hooks for chain_job.sh (verl).
#
# Sourced by chain_job.sh on every iteration (including initial launch and every
# sbatch resubmission). Must define:
#   - shell variables: CHAIN_WORKDIR, CHAIN_SBATCH_ARGS (array)
#                      (CHAIN_ROOT and LOG_ROOT optional; default under CHAIN_WORKDIR)
#   - shell function:  chain_exec_target  (takes $1=target_script, runs it)
#
# Available to chain_exec_target (exported by chain_job.sh before calling):
#   RUN_ID, CHAIN_DIR, CHAIN_FLAG_FILE, JOB_END_TS

CHAIN_WORKDIR="/work/hdd/bbsg/twei2/rl/verl"

CHAIN_SBATCH_ARGS=(
    --account=bbsg-dtai-gh
    --partition=ghx4-interactive
    --nodes=1
    --ntasks-per-node=1
    --cpus-per-task=64
    --mem=0
)

: "${VERL:=apptainer exec --nv --bind /work,/u /work/hdd/bbsg/twei2/rl/torch2501.sif}"
export VERL

chain_exec_target() {
    local target="$1"
    srun $VERL bash -c "source /work/hdd/bbsg/twei2/rl/verl_container_rc.sh && \
                        cd ${CHAIN_WORKDIR} && \
                        RUN_ID='${RUN_ID}' CHAIN_DIR='${CHAIN_DIR}' CHAIN_FLAG_FILE='${CHAIN_FLAG_FILE}' \
                        MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP='${JOB_END_TS}' \
                        PYTHONUNBUFFERED=1 \
                        stdbuf -oL -eL bash '${target}'"
}
