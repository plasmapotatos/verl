#!/bin/bash
# Diagnostic target script for chain_job.sh — verifies the ESI timestamp
# env var is plumbed through srun + container correctly.
# Usage: ./chain_job.sh experiments/rl/esi_probe.sh
# Then check: cat logs/chain/esi_probe/iter_*.out

echo "=== ESI PROBE ==="
echo "MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP=${MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP:-<UNSET>}"
echo "now=$(date +%s)"

if [[ -n "${MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP:-}" ]]; then
    REMAIN=$(( MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP - $(date +%s) ))
    echo "remaining seconds: $REMAIN"
    if [[ "$REMAIN" -gt 0 ]]; then
        echo "PASS: env var is set and in the future"
    else
        echo "WARN: env var is set but already in the past (remaining=$REMAIN)"
    fi
else
    echo "FAIL: env var is not set — verl ESI-save will not work"
fi

echo ""
echo "RUN_ID=${RUN_ID:-<UNSET>}"
echo "CHAIN_DIR=${CHAIN_DIR:-<UNSET>}"
echo "CHAIN_FLAG_FILE=${CHAIN_FLAG_FILE:-<UNSET>}"

# Signal completion so the chain doesn't resubmit
if [[ -n "${CHAIN_FLAG_FILE:-}" ]]; then
    touch "$CHAIN_FLAG_FILE"
    echo "Touched complete flag: $CHAIN_FLAG_FILE"
else
    echo "WARN: CHAIN_FLAG_FILE not set, chain will resubmit"
fi

echo "=== DONE ==="
