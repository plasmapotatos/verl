#!/bin/bash
# Probe: records start timestamp and exits immediately.
set -u
OUT_DIR="${QB_OUT_DIR:?QB_OUT_DIR not set}"
echo "start_ts=$(date +%s)" > "${OUT_DIR}/start_${SLURM_JOB_ID}.txt"
echo "start_human=$(date -Iseconds)" >> "${OUT_DIR}/start_${SLURM_JOB_ID}.txt"
echo "hostname=$(hostname)" >> "${OUT_DIR}/start_${SLURM_JOB_ID}.txt"
exit 0
