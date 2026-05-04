#!/bin/bash
# Pretty-print recent completed chain_job experiments from completed_experiments.tsv.
#
# Usage: list_completed.sh [-n N] [--log-root DIR]

set -uo pipefail

SELF_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
LOG_ROOT="${SELF_DIR}/logs/chain"
N=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n) N="$2"; shift 2 ;;
        --log-root) LOG_ROOT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [-n N] [--log-root DIR]" >&2
            exit 1 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

LOG="${LOG_ROOT}/completed_experiments.tsv"
[[ -f "$LOG" ]] || { echo "No completion log yet at $LOG" >&2; exit 0; }

{
    head -n 1 "$LOG"
    tail -n +2 "$LOG" | tail -n "$N"
} | column -t -s $'\t'
