#!/usr/bin/env bash
set -euo pipefail

# Expand SimpleQA rich-QA RL train parquet into n=5 rows per source QA
# (1 original + 4 paraphrased questions sharing the same answer).
#
# Uses the shared augment CLI so OpenAI calls fan out across BATCH_SIZE threads.

REPO_ROOT=/work/hdd/bbsg/twei2/rl/verl

INPUT=${INPUT:-$REPO_ROOT/data/simpleqa/partition/rich_qa/rl/train.parquet}
OUTPUT=${OUTPUT:-$REPO_ROOT/data/simpleqa/partition/rich_qa/rl/train_n5_paraphrase.parquet}
# Total rows per source = 1 original (from --mix_original) + N_PARAPHRASES paraphrases.
N_PARAPHRASES=${N_PARAPHRASES:-4}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-32}

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Export it before running." >&2
  exit 1
fi

cd "$REPO_ROOT"

python -m verl.augment.cli \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --method simpleqa_n_paraphrase \
  --n "$N_PARAPHRASES" \
  --mix_original \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  "$@"
