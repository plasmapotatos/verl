#!/bin/bash
# Copy a file or folder to /tmp/shared/ for sharing with coworkers,
# stripping out large model weight files (.pt, .safetensors, .bin, .ckpt).
#
# Usage: ./scripts/share_experiment.sh <path> [dest_base]
#   dest_base defaults to /tmp/shared

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <file_or_folder> [dest_base]"
    echo "Example: $0 /work/hdd/bbsg/twei2/rl/verl/outputs/sft/my_experiment"
    echo "         $0 /work/hdd/bbsg/twei2/rl/verl/scripts/some_file.py"
    exit 1
fi

SRC="$(realpath "$1")"
DEST_BASE="${2:-/tmp/shared}"

if [ ! -e "$SRC" ]; then
    echo "Error: $SRC does not exist"
    exit 1
fi

if [ -f "$SRC" ]; then
    # Single file
    DEST="$DEST_BASE/$(basename "$SRC")"
    echo "Source:      $SRC"
    echo "Destination: $DEST"
    echo ""
    mkdir -p "$DEST_BASE"
    cp "$SRC" "$DEST"
    chmod a+r "$DEST"
    FINAL_SIZE=$(du -sh "$DEST" | cut -f1)
    echo "Done! Shared at: $DEST ($FINAL_SIZE)"
    echo "Readable by all users."
else
    # Directory
    FOLDER_NAME="$(basename "$SRC")"
    DEST="$DEST_BASE/$FOLDER_NAME"
    echo "Source:      $SRC"
    echo "Destination: $DEST"
    echo ""

    # Excluded extensions (large model files)
    EXCLUDES=(
        --exclude='*.pt'
        --exclude='*.safetensors'
        --exclude='*.bin'
        --exclude='*.ckpt'
    )

    # Show what will be excluded
    echo "Excluding file types: .pt .safetensors .bin .ckpt"
    EXCLUDED_SIZE=$(find "$SRC" -type f \( -name "*.pt" -o -name "*.safetensors" -o -name "*.bin" -o -name "*.ckpt" \) -exec du -cb {} + 2>/dev/null | tail -1 | cut -f1)
    EXCLUDED_SIZE=${EXCLUDED_SIZE:-0}
    echo "Skipping ~$(numfmt --to=iec "$EXCLUDED_SIZE") of model weights"
    echo ""

    # Copy with rsync, excluding heavy files
    mkdir -p "$DEST"
    rsync -av "${EXCLUDES[@]}" "$SRC/" "$DEST/"

    # Make everything world-readable
    chmod -R a+rX "$DEST"

    FINAL_SIZE=$(du -sh "$DEST" | cut -f1)
    echo ""
    echo "Done! Shared at: $DEST ($FINAL_SIZE)"
    echo "Readable by all users."
fi
