#!/usr/bin/env bash
# Build a chain-of-thought variant of train_eval_k3.parquet by appending a
# "think step by step" instruction to the user prompt.
#
# verl.trainer.main_generation reads from data.prompt_key=prompt, where each
# row is a list of chat messages: [{"role": "user", "content": <question>}].
# We modify both that content and the `question` column so eval/judge logs
# reflect what the model actually saw.

set -euo pipefail

SFT="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose_mixed_k12_smoke/sft"
SRC="$SFT/train_eval_k3.parquet"
DST="$SFT/train_eval_k3_cot.parquet"

python - <<PY
import pandas as pd

SUFFIX = "\n\nThink step by step, then give your final answer."

df = pd.read_parquet("$SRC")

def patch_prompt(prompt):
    out = []
    for msg in prompt:
        m = dict(msg)
        if m.get("role") == "user":
            m["content"] = m["content"].rstrip() + SUFFIX
        out.append(m)
    return out

df["prompt"] = df["prompt"].apply(patch_prompt)
df["question"] = df["question"].apply(lambda q: q.rstrip() + SUFFIX)
df.to_parquet("$DST", index=False)
print(f"wrote {len(df)} rows -> $DST")
print("sample prompt:", df.iloc[0]["prompt"][0]["content"][:300])
PY
