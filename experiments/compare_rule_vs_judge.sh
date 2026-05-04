#!/usr/bin/env bash
# Compare rule-based vs LLM judge grading on richqa_grpo step 680 generations.
set -euo pipefail

GENERATIONS_DIR="/work/hdd/bbsg/twei2/rl/verl/outputs/rl/richqa_grpo/binary/global_step_680/generations"
OUT_DIR="${GENERATIONS_DIR}/eval_comparison"
mkdir -p "$OUT_DIR"

# Val split
VAL_PARQUET="${GENERATIONS_DIR}/binary_global_step_680__on_val.parquet"
# Train-eval split
TRAIN_PARQUET="${GENERATIONS_DIR}/binary_global_step_680__on_train_eval.parquet"

EVAL_CMD="python -m verl.eval.cli --dataset simpleqa"

echo "=== Rule-based grading (val) ==="
$EVAL_CMD \
    --input "$VAL_PARQUET" \
    --output "$OUT_DIR/val_rule.json" \
    --workers 16

echo ""
echo "=== LLM judge grading (val) ==="
$EVAL_CMD \
    --input "$VAL_PARQUET" \
    --output "$OUT_DIR/val_judge.json" \
    --use-judge \
    --judge-model gpt-4o-mini \
    --workers 16

echo ""
echo "=== Rule-based grading (train_eval) ==="
$EVAL_CMD \
    --input "$TRAIN_PARQUET" \
    --output "$OUT_DIR/train_eval_rule.json" \
    --workers 16

echo ""
echo "=== LLM judge grading (train_eval) ==="
$EVAL_CMD \
    --input "$TRAIN_PARQUET" \
    --output "$OUT_DIR/train_eval_judge.json" \
    --use-judge \
    --judge-model gpt-4o-mini \
    --workers 16

echo ""
echo "Results written to $OUT_DIR"
echo ""
echo "=== Metrics summary ==="
python - <<'EOF'
import json, pathlib

out = pathlib.Path("/work/hdd/bbsg/twei2/rl/verl/outputs/rl/richqa_grpo/binary/global_step_680/generations/eval_comparison")
for name in ("val_rule", "val_judge", "train_eval_rule", "train_eval_judge"):
    p = out / f"{name}.json"
    if not p.exists():
        print(f"{name}: missing")
        continue
    data = json.loads(p.read_text())
    metrics = data.get("metrics", {})
    print(f"{name}: {metrics}")
EOF
