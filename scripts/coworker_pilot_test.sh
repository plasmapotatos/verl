#!/usr/bin/env bash
# Pilot test for a coworker running twei2's experiments out of
# /work/hdd/bbsg/twei2/rl/verl/. Run this as YOUR user (not twei2) on a
# Delta login node BEFORE launching any chain_job.sh. It checks group
# membership, SLURM account access, file permissions, container readability,
# wandb login, and (optionally) submits a tiny 1-GPU sbatch that imports
# torch+verl inside the container.
#
# Usage:
#   bash coworker_pilot_test.sh           # local checks only
#   bash coworker_pilot_test.sh --sbatch  # also submit ~10min 1-GPU smoke job

set -uo pipefail

WORKDIR="/work/hdd/bbsg/twei2/rl/verl"
SIF="/work/hdd/bbsg/twei2/rl/torch2501.sif"
GROUP="delta_bbsg"
ACCOUNT="bbsg-dtai-gh"
PARTITION="ghx4-interactive"
RC="/work/hdd/bbsg/twei2/rl/verl_container_rc.sh"

DO_SBATCH=0
[[ "${1:-}" == "--sbatch" ]] && DO_SBATCH=1

PASS=0
FAIL=0
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$*"; }
hdr()  { printf '\n=== %s ===\n' "$*"; }

hdr "Identity"
echo "  user: $USER   host: $(hostname)"

hdr "Linux group membership ($GROUP)"
if id -nG "$USER" | tr ' ' '\n' | grep -qx "$GROUP"; then
    ok "in group $GROUP"
else
    bad "not in group $GROUP — ask twei2/admin to add you (group rwx ACLs depend on it)"
fi

hdr "SLURM account ($ACCOUNT)"
if command -v sacctmgr >/dev/null 2>&1; then
    if sacctmgr -nP show assoc user="$USER" format=account 2>/dev/null \
            | grep -qx "$ACCOUNT"; then
        ok "associated with account $ACCOUNT"
    else
        bad "no association with $ACCOUNT — sbatch will be rejected"
    fi
else
    warn "sacctmgr not on PATH; skipping account check"
fi

hdr "Container image"
if [[ -r "$SIF" ]]; then
    ok "readable: $SIF"
else
    bad "cannot read $SIF"
fi

hdr "Write access into twei2's tree"
for d in "$WORKDIR/outputs" "$WORKDIR/outputs/sft" \
         "$WORKDIR/chain_state" "$WORKDIR/logs/chain"; do
    probe="$d/.permtest_${USER}_$$"
    if ( : > "$probe" ) 2>/dev/null; then
        rm -f "$probe"
        ok "writable: $d"
    else
        bad "NOT writable: $d  (group ACL or membership issue)"
    fi
done

hdr "Container rc + apptainer exec"
if command -v apptainer >/dev/null 2>&1; then
    if [[ -r "$RC" ]]; then
        ok "rc readable: $RC"
    else
        bad "rc not readable: $RC"
    fi
    if apptainer exec --bind /work,/u "$SIF" \
           python -c "import sys; print('python', sys.version.split()[0])" \
           >/tmp/pilot_py.$$ 2>&1; then
        ok "apptainer exec python works ($(cat /tmp/pilot_py.$$))"
    else
        bad "apptainer exec python failed:"
        sed 's/^/       /' /tmp/pilot_py.$$
    fi
    rm -f /tmp/pilot_py.$$
else
    bad "apptainer not on PATH"
fi

hdr "wandb + OpenAI credentials (inside container, after sourcing rc)"
# These are normally exported by /work/hdd/bbsg/twei2/rl/venv/bin/activate
# (sourced via $RC), so check them in that exact context — not the host shell.
if apptainer exec --bind /work,/u "$SIF" bash -c "
        source '$RC' >/dev/null 2>&1
        rc=0
        if [[ -n \"\${WANDB_API_KEY:-}\" ]]; then
            echo 'wandb_ok'
        else
            echo 'wandb_missing'; rc=1
        fi
        if [[ -n \"\${OPENAI_API_KEY:-}\" ]]; then
            echo 'openai_ok'
        else
            echo 'openai_missing'
        fi
        exit \$rc
    " >/tmp/pilot_keys.$$ 2>&1; then
    grep -q wandb_ok  /tmp/pilot_keys.$$ && ok "WANDB_API_KEY exported by rc/venv activate"
    grep -q openai_ok /tmp/pilot_keys.$$ && ok "OPENAI_API_KEY exported by rc/venv activate"
else
    bad "WANDB_API_KEY not set after sourcing $RC inside the container"
    sed 's/^/       /' /tmp/pilot_keys.$$
fi
rm -f /tmp/pilot_keys.$$

hdr "Existing chain collisions (your user only)"
EXISTING="$(squeue -h -u "$USER" -o '%j %i %T' 2>/dev/null | grep '^chain_' || true)"
if [[ -z "$EXISTING" ]]; then
    ok "no chain_* jobs queued under $USER"
else
    warn "you already have chain jobs queued:"
    echo "$EXISTING" | sed 's/^/       /'
fi

hdr "Summary"
echo "  passed: $PASS    failed: $FAIL"
if [[ "$FAIL" -gt 0 ]]; then
    echo
    echo "Fix the FAILs above before running run_chain.sh."
    exit 1
fi

if [[ "$DO_SBATCH" -ne 1 ]]; then
    echo
    echo "Local checks passed. Re-run with --sbatch to submit a 1-GPU 10-min"
    echo "smoke job that imports torch + verl inside the container."
    exit 0
fi

hdr "Submitting 1-GPU smoke sbatch"
LOG="/tmp/pilot_smoke_${USER}_$$.out"
JOB_ID="$(sbatch --parsable \
    --account="$ACCOUNT" --partition="$PARTITION" \
    --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 \
    --time=00:10:00 \
    --job-name="pilot_${USER}" \
    --output="$LOG" --error="$LOG" \
    --wrap "apptainer exec --nv --bind /work,/u $SIF \
            bash -c 'source $RC && \
              python -c \"import torch, verl; \
                print(\\\"torch\\\", torch.__version__); \
                print(\\\"cuda_avail\\\", torch.cuda.is_available()); \
                print(\\\"gpu_count\\\", torch.cuda.device_count()); \
                print(\\\"verl\\\", getattr(verl, \\\"__version__\\\", \\\"?\\\"))\"'" \
    2>&1)"
RC_SUB=$?
if [[ "$RC_SUB" -ne 0 || -z "$JOB_ID" ]]; then
    bad "sbatch submission failed: $JOB_ID"
    exit 1
fi
ok "submitted job $JOB_ID  (log: $LOG)"
echo
echo "Watch with:   squeue -j $JOB_ID"
echo "Tail log:     tail -f $LOG"
echo "When the job ends, cat the log — you should see torch+cuda+verl info"
echo "and no Python tracebacks. Then you're good to launch run_chain.sh."
