#!/usr/bin/env bash
# Remove chain_state/<RUN_ID> and logs/chain/<RUN_ID> dirs for any RUN_ID
# that does NOT currently have a SLURM job named chain_<RUN_ID>.
#
# Usage:
#   ./clean_chain_state.sh            # dry-run: list what would be deleted
#   ./clean_chain_state.sh --yes      # actually delete

set -uo pipefail

SELF_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
CHAIN_ROOT="${SELF_DIR}/chain_state"
LOG_ROOT="${SELF_DIR}/logs/chain"

APPLY=0
[[ "${1:-}" == "--yes" || "${1:-}" == "-y" ]] && APPLY=1

mapfile -t ACTIVE < <(
    squeue -h -u "$USER" -o '%j' 2>/dev/null \
        | awk '/^chain_/ { sub(/^chain_/, ""); print }' \
        | sort -u
)

echo "[clean] Active chain RUN_IDs (${#ACTIVE[@]}):"
for r in "${ACTIVE[@]}"; do echo "    $r"; done
[[ "${#ACTIVE[@]}" -eq 0 ]] && echo "    (none)"
echo

is_active() {
    local name="$1"
    for a in "${ACTIVE[@]}"; do [[ "$a" == "$name" ]] && return 0; done
    return 1
}

collect_victims() {
    local root="$1"
    [[ -d "$root" ]] || return 0
    for d in "$root"/*/; do
        [[ -d "$d" ]] || continue
        local name
        name="$(basename "$d")"
        is_active "$name" || echo "$d"
    done
}

mapfile -t STATE_VICTIMS < <(collect_victims "$CHAIN_ROOT")
mapfile -t LOG_VICTIMS   < <(collect_victims "$LOG_ROOT")

if [[ "${#STATE_VICTIMS[@]}" -eq 0 && "${#LOG_VICTIMS[@]}" -eq 0 ]]; then
    echo "[clean] Nothing to delete."
    exit 0
fi

echo "[clean] Would remove ${#STATE_VICTIMS[@]} chain_state dir(s):"
for d in "${STATE_VICTIMS[@]}"; do echo "    $d"; done
echo "[clean] Would remove ${#LOG_VICTIMS[@]} logs/chain dir(s):"
for d in "${LOG_VICTIMS[@]}"; do echo "    $d"; done

if [[ "$APPLY" -ne 1 ]]; then
    echo
    echo "[clean] Dry-run. Re-run with --yes to delete."
    exit 0
fi

echo
echo "[clean] Deleting..."
for d in "${STATE_VICTIMS[@]}" "${LOG_VICTIMS[@]}"; do
    rm -rf -- "$d" && echo "    removed $d"
done
echo "[clean] Done."
