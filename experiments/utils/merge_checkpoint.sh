#!/usr/bin/env bash
set -e

# Merge FSDP checkpoint to HF format
# Usage: merge_checkpoint.sh <ckpt_dir>

CKPT_DIR="$1"
if [[ -z "$CKPT_DIR" ]]; then
    echo "Usage: $0 <ckpt_dir>"
    exit 1
fi

VERL_DIR="${VERL_DIR:-/work/hdd/bbsg/twei2/rl/verl}"
# Normalize checkpoint path so absolute paths are not prefixed twice.
if [[ "$CKPT_DIR" = /* ]]; then
    CKPT_PATH="$CKPT_DIR"
elif [[ -d "$CKPT_DIR" ]]; then
    CKPT_PATH="$(cd "$CKPT_DIR" && pwd)"
else
    CKPT_PATH="$VERL_DIR/$CKPT_DIR"
fi

if [[ ! -d "$CKPT_PATH" ]]; then
    echo "Checkpoint directory not found: $CKPT_PATH"
    exit 1
fi

LOCAL_DIR="$CKPT_PATH"
if [[ -d "$CKPT_PATH/actor" ]]; then
    LOCAL_DIR="$CKPT_PATH/actor"
fi

MERGED_DIR="$CKPT_PATH/merged_hf_model"
if [[ -d "$MERGED_DIR" ]]; then
    if [[ -f "$MERGED_DIR/config.json" ]]; then
        echo "Merged model already exists and is valid: $MERGED_DIR"
        exit 0
    else
        echo "Merged model directory exists but appears incomplete (no config.json) — removing and re-merging: $MERGED_DIR"
        rm -rf "$MERGED_DIR"
    fi
fi

echo "Merging $LOCAL_DIR -> $MERGED_DIR"
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$LOCAL_DIR" \
    --target_dir "$MERGED_DIR"

# Copy aux files if Qwen2.5-VL
if [[ -f "$MERGED_DIR/config.json" ]]; then
    MODEL_TYPE=$(python3 -c "import json; print(json.load(open('$MERGED_DIR/config.json'))['model_type'])")
    if [[ "$MODEL_TYPE" == "qwen2_vl" || "$MODEL_TYPE" == "qwen2_5_vl" ]]; then
        echo "Copying aux files for $MODEL_TYPE"
        # Assume HF_AUX_SRC_DIR is set or default
        HF_AUX_SRC_DIR="${HF_AUX_SRC_DIR:-/work/hdd/bbsg/twei2/rl/verl/configs/hf_aux_files}"
        cp -f "$HF_AUX_SRC_DIR/preprocessor_config.json" "$MERGED_DIR/" 2>/dev/null || true
        cp -f "$HF_AUX_SRC_DIR/chat_template.json" "$MERGED_DIR/" 2>/dev/null || true
    fi
fi

echo "Merged to $MERGED_DIR"
