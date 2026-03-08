#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint_dir> [output_dir]"
  exit 1
fi

CKPT_DIR="$1"
OUT_DIR="${2:-$CKPT_DIR/merged_hf_model}"

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "Checkpoint dir not found: $CKPT_DIR"
  exit 1
fi

LOCAL_DIR="$CKPT_DIR"
if [[ -d "$CKPT_DIR/actor" ]]; then
  if ! ls "$CKPT_DIR"/model_world_size_* >/dev/null 2>&1; then
    LOCAL_DIR="$CKPT_DIR/actor"
  fi
fi

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$LOCAL_DIR" \
  --target_dir "$OUT_DIR"

CONFIG_PATH="$OUT_DIR/config.json"
if [[ -f "$CONFIG_PATH" ]]; then
  if grep -qiE "qwen2_5_vl|qwen2.5-vl" "$CONFIG_PATH"; then
    AUX_DIR="/work/hdd/bbsg/twei2/rl/verl/configs/hf_aux_files"
    PREPROC_SRC="$AUX_DIR/preprocessor_config.json"
    CHAT_TEMPLATE_SRC="$AUX_DIR/chat_template.json"

    if [[ -f "$PREPROC_SRC" ]]; then
      cp -f "$PREPROC_SRC" "$OUT_DIR/preprocessor_config.json"
    else
      echo "Warning: missing $PREPROC_SRC"
    fi

    if [[ -f "$CHAT_TEMPLATE_SRC" ]]; then
      cp -f "$CHAT_TEMPLATE_SRC" "$OUT_DIR/chat_template.json"
    else
      echo "Warning: missing $CHAT_TEMPLATE_SRC"
    fi
  fi
fi
