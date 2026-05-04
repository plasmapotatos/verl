#!/bin/bash
# Open the chain_job log for a given SLURM job id in the current VS Code window.
#
# Usage: open_log.sh <jobid> [--err] [--log-root DIR]
#
# Looks under ${LOG_ROOT}/*/iter_<jobid>.out (or .err with --err).
# LOG_ROOT defaults to $(dirname $0)/logs/chain to match chain_job.sh.

set -uo pipefail

SELF_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
LOG_ROOT="${SELF_DIR}/logs/chain"
EXT="out"
JOBID=""

usage() {
    cat >&2 <<EOF
Usage: $0 <jobid> [--err] [--log-root DIR]

Opens the matching iter_<jobid>.{out,err} log for a chain_job SLURM job
in the current VS Code window (via 'code -r').
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --err) EXT="err"; shift ;;
        --out) EXT="out"; shift ;;
        --log-root) [[ $# -ge 2 ]] || usage; LOG_ROOT="$2"; shift 2 ;;
        --log-root=*) LOG_ROOT="${1#--log-root=}"; shift ;;
        -h|--help) usage ;;
        -*) echo "Unknown option: $1" >&2; usage ;;
        *)
            [[ -z "$JOBID" ]] || usage
            JOBID="$1"
            shift
            ;;
    esac
done

[[ -n "$JOBID" ]] || usage
[[ "$JOBID" =~ ^[0-9]+$ ]] || { echo "ERROR: jobid must be numeric, got '$JOBID'" >&2; exit 1; }
[[ -d "$LOG_ROOT" ]] || { echo "ERROR: log root not found: $LOG_ROOT" >&2; exit 1; }

mapfile -t MATCHES < <(find "$LOG_ROOT" -maxdepth 2 -type f -name "iter_${JOBID}.${EXT}" 2>/dev/null)

if [[ ${#MATCHES[@]} -eq 0 ]]; then
    echo "No iter_${JOBID}.${EXT} found under ${LOG_ROOT}/" >&2
    exit 1
fi

if [[ ${#MATCHES[@]} -gt 1 ]]; then
    echo "Multiple matches (using first):" >&2
    printf '  %s\n' "${MATCHES[@]}" >&2
fi

TARGET="${MATCHES[0]}"
echo "Opening: $TARGET"

# Refresh VSCODE_IPC_HOOK_CLI: remote-ssh creates a new socket per session, so
# a stale terminal (tmux, old ssh) will hold a dead socket path. Probe all live
# sockets and use the first that accepts `code --status`.
pick_live_ipc() {
    local sock
    while IFS= read -r sock; do
        [[ -S "$sock" ]] || continue
        if VSCODE_IPC_HOOK_CLI="$sock" timeout 2 code --status >/dev/null 2>&1; then
            echo "$sock"
            return 0
        fi
    done < <(ls -t /run/user/"$(id -u)"/vscode-ipc-*.sock 2>/dev/null)
    return 1
}

if ! VSCODE_IPC_HOOK_CLI="${VSCODE_IPC_HOOK_CLI:-}" timeout 2 code --status >/dev/null 2>&1; then
    if LIVE_SOCK="$(pick_live_ipc)"; then
        export VSCODE_IPC_HOOK_CLI="$LIVE_SOCK"
        echo "[open_log] using live VS Code socket: $LIVE_SOCK"
    else
        echo "ERROR: no live VS Code IPC socket found. Open a terminal inside VS Code and retry." >&2
        exit 1
    fi
fi

code -r "$TARGET"
