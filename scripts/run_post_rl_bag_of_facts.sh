#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Defaults from the requested experiment setup.
SFT_PATH="${SFT_PATH:-outputs/sft/simpleqa_factual_anchor_sft_smoketest/sft_lr1.5e-4_epmax30_seed1/global_step_570/generations/sft_lr1.5e-4_epmax30_seed1_global_step_570__on_train_eval_eval.json}"
RL_GEN_DIR="${RL_GEN_DIR:-outputs/rl/simpleqa_factual_anchor_grpo_smoketest/binary/global_step_300/generations}"

TRAIN_EVAL_PATH="${TRAIN_EVAL_PATH:-$RL_GEN_DIR/binary_global_step_300__on_train_eval_eval.json}"
VAL_PATH="${VAL_PATH:-$RL_GEN_DIR/binary_global_step_300__on_val_eval.json}"

OUT_DIR="${OUT_DIR:-$RL_GEN_DIR/bag_of_facts_analysis}"
mkdir -p "$OUT_DIR"

# Use the repo's required execution environment by default.
PYTHON_CMD="${PYTHON_CMD:-apptainer exec /work/hdd/bbsg/twei2/rl/torch2501.sif python}"

echo "[1/2] Running bag-of-facts on RL train-eval"
$PYTHON_CMD analysis_bag_of_facts.py \
  --sft "$SFT_PATH" \
  --origqa "$TRAIN_EVAL_PATH" \
  | tee "$OUT_DIR/train_eval.txt"

echo "[2/2] Running bag-of-facts on RL val"
$PYTHON_CMD analysis_bag_of_facts.py \
  --sft "$SFT_PATH" \
  --origqa "$VAL_PATH" \
  | tee "$OUT_DIR/val.txt"

echo
echo "Done. Outputs written to: $OUT_DIR"
echo "  - $OUT_DIR/train_eval.txt"
echo "  - $OUT_DIR/val.txt"
