#!/bin/bash
# Sync artifacts from the team shared directory into this repo's shared/ folder.
#
# Defaults:
#   Source:      /work/hdd/bbsg/shared
#   Destination: <repo_root>/shared
#
# Usage:
#   ./scripts/sync_shared.sh [--delete] [--dry-run] [source_dir] [dest_dir]
#
# Examples:
#   ./scripts/sync_shared.sh
#   ./scripts/sync_shared.sh --dry-run
#   ./scripts/sync_shared.sh --delete
#   ./scripts/sync_shared.sh /work/hdd/bbsg/shared /work/hdd/bbsg/twei2/rl/verl/shared

set -euo pipefail

DELETE_MODE=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --delete)
            DELETE_MODE=1
            shift
            ;;
        --dry-run|-n)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            sed -n '1,30p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use --help for usage."
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR/..")"

SRC="${1:-/work/hdd/bbsg/shared}"
DEST="${2:-$REPO_ROOT/shared}"

SRC="$(realpath "$SRC")"

if [ ! -d "$SRC" ]; then
    echo "Error: source directory does not exist: $SRC"
    exit 1
fi

mkdir -p "$DEST"

echo "Source:      $SRC"
echo "Destination: $DEST"
echo ""

RSYNC_ARGS=(
    -avh
    --human-readable
    --info=progress2
    --exclude='*.pt'
    --exclude='*.safetensors'
    --exclude='*.bin'
    --exclude='*.ckpt'
)

if [ "$DELETE_MODE" -eq 1 ]; then
    RSYNC_ARGS+=(--delete)
    echo "Mode: mirror sync (includes deletions in destination)"
else
    echo "Mode: additive sync (no deletions in destination)"
fi

if [ "$DRY_RUN" -eq 1 ]; then
    RSYNC_ARGS+=(--dry-run)
    echo "Dry run: enabled"
fi

echo "Excluding file types: .pt .safetensors .bin .ckpt"
echo ""

rsync "${RSYNC_ARGS[@]}" "$SRC/" "$DEST/"

echo ""
echo "Sync complete."