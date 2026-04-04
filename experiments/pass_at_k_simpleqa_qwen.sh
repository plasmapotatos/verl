#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_DATA="$REPO_ROOT/data/simpleqa/data.parquet"
MODEL="Qwen/Qwen2.5-3B-Instruct"
OUTPUT_ROOT="$REPO_ROOT/outputs/simpleqa_qwen_pass_at_k"
K_VALUES=(1 16 32 64 128)

if [[ ! -f "$EVAL_DATA" ]]; then
  echo "ERROR: Eval parquet not found: $EVAL_DATA"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
for k in "${K_VALUES[@]}"; do
  RUN_DIR="$OUTPUT_ROOT/k_$k"
  mkdir -p "$RUN_DIR"

  echo "Running pass@k with k=$k"
  python3 "$REPO_ROOT/scripts/pass_at_k.py" \
    --checkpoint "$MODEL" \
    --dataset simpleqa \
    --eval-data "$EVAL_DATA" \
    --output-dir "$RUN_DIR" \
    --top-k "$k" \
    --use-judge

  SUMMARY_PATH="$RUN_DIR/pass@k/pass_at_k_${k}.json"
  if [[ ! -f "$SUMMARY_PATH" ]]; then
    echo "ERROR: Missing summary for k=$k"
    exit 1
  fi
done

K_VALUES_PY="[$(IFS=,; echo "${K_VALUES[*]}")]"
PLOT_PATH="$OUTPUT_ROOT/pass_at_k_plot.png"
python3 - <<PY
import json
from pathlib import Path
import matplotlib.pyplot as plt

output_root = Path("$OUTPUT_ROOT")
k_values = ${K_VALUES_PY}
pass_rates = []
for k in k_values:
    summary_path = output_root / f"k_{k}" / "pass@k" / f"pass_at_k_{k}.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    pass_rates.append(data.get("pass_at_k", {}).get("pass_at_k", 0.0))

plt.figure(figsize=(6, 4))
plt.plot(k_values, pass_rates, marker="o", color="#1d4ed8")
plt.title("Qwen-3B pass@k on SimpleQA")
plt.xlabel("k")
plt.ylabel("pass@k")
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
output_root.mkdir(parents=True, exist_ok=True)
plt.savefig("$PLOT_PATH")
PY
echo "pass@k graph written to $PLOT_PATH"