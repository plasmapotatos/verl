"""Step 3 (after combine_qa): build SFT and RL datasets for the mixed k=1/k=2 smoketest.

New layout (same ids across SFT / RL-train / RL-val; k=2 combos are disjoint per id):

  Per id with g sub-Q/A, sort combined-k=2 rows by draw_idx and partition:
    [0 : g]                  -> SFT seen pool         (sft/train k=2 half)
    [g]                      -> SFT held-out k=2 eval (sft/train_eval_k2_unique, 1 per id)
    [g+1 : g+1+M]            -> RL train pool         (rl/train, M per id)
    [g+1+M]                  -> RL val pool           (rl/val candidate, 1 per id)
    [g+1+M+1 :]              -> unused

SFT outputs (data/.../decompose_mixed_k12_smoke/sft/):
  train.parquet                — k=1 sub-Q/A rows + SFT-seen k=2 rows.
  train_eval_k1.parquet        — 1 k=1 row per id.
  train_eval_k2.parquet        — 1 SFT-seen k=2 row per id.
  train_eval_k2_unique.parquet — 1 held-out k=2 row per id (in neither SFT nor RL).
  train_eval_origqa.parquet    — 1 origqa row per id.

RL outputs (data/.../decompose_mixed_k12_smoke/rl/):
  train.parquet                — RL train pool: M k=2 combos per id, all 688 ids.
  train_eval.parquet           — random subsample of train, size ≤ EVAL_SIZE.
  val.parquet                  — 1 held-out k=2 per id, subsampled to size ≤ EVAL_SIZE.
  eval_k1.parquet              — 1 k=1 per id, subsampled to ≤ EVAL_SIZE (diagnostic).
  eval_k2_seen.parquet         — 1 SFT-seen k=2 per id, subsampled to ≤ EVAL_SIZE.
  eval_origqa.parquet          — 1 origqa per id, subsampled to ≤ EVAL_SIZE.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose_mixed_k12_smoke")
INTER = ROOT / "_intermediate"
SFT_OUT = ROOT / "sft"
RL_OUT = ROOT / "rl"

ORIGQA_SOURCES = [
    "/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose_and_richqa_capped/rl/train.parquet",
    "/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose_and_richqa_capped/rl/val.parquet",
]


def _normalize_metadata(template_row: dict) -> dict:
    md = template_row.get("metadata")
    if isinstance(md, dict):
        out = {}
        for k, v in md.items():
            out[k] = v.tolist() if isinstance(v, np.ndarray) else v
        return out
    return md


def build_k2_row(template_subq_row: pd.Series, combined: dict, idx_counter: int) -> dict:
    template = template_subq_row.to_dict()
    metadata = _normalize_metadata(template)
    question = combined["question"]
    answer = combined["answer"]
    gid = combined["id"]
    extra_info = {
        "answer": answer,
        "augmentation": {
            "method": "combine_qa",
            "original_sample_id": gid,
            "draw_idx": int(combined["draw_idx"]),
            "k": 2,
            "source_subq_indices": list(combined["source_subq_indices"]),
            "source_questions": list(combined["source_questions"]),
            "source_answers": list(combined["source_answers"]),
        },
        "index": idx_counter,
        "metadata": metadata,
        "original_answer": answer,
        "original_question": question,
        "question": question,
        "sample_id": gid,
        "split": "train",
    }
    return {
        "id": gid,
        "question": question,
        "answer": answer,
        "metadata": metadata,
        "data_source": "simpleqa",
        "prompt": [{"content": question, "role": "user"}],
        "ability": "general",
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": extra_info,
    }


def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl_train_per_id", type=int, default=6,
                    help="M: number of held-out k=2 combos used as RL train data per id.")
    ap.add_argument("--eval_size", type=int, default=256,
                    help="Cap for train_eval / val / per-slice eval files.")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    M = args.rl_train_per_id

    subqas = pd.read_parquet(INTER / "decompose_mid_subqas.parquet")
    combined = pd.read_parquet(INTER / "combined_k2.parquet")
    origqa_full = pd.concat(
        [pd.read_parquet(p) for p in ORIGQA_SOURCES], ignore_index=True
    )

    selected_ids = sorted(subqas["id"].unique().tolist())
    sizes = subqas.groupby("id").size()

    SFT_OUT.mkdir(parents=True, exist_ok=True)
    RL_OUT.mkdir(parents=True, exist_ok=True)

    # Per-id partition of combined k=2 by draw_idx.
    seen_records: list[dict] = []
    sft_unique_records: list[dict] = []
    rl_train_records: list[dict] = []
    rl_val_records: list[dict] = []
    counter = 0
    insufficient_ids = []
    for gid, sub_combined in combined.groupby("id"):
        g = int(sizes.loc[gid])
        sub_combined = sub_combined.sort_values("draw_idx").reset_index(drop=True)
        template_row = subqas[subqas["id"] == gid].iloc[0]
        # required positions: [0:g], [g], [g+1:g+1+M], [g+1+M]
        needed = g + 1 + M + 1
        if len(sub_combined) < needed:
            insufficient_ids.append((gid, g, len(sub_combined), needed))
        for i, row in sub_combined.iterrows():
            built = build_k2_row(template_row, row.to_dict(), counter)
            counter += 1
            if i < g:
                seen_records.append(built)
            elif i == g:
                sft_unique_records.append(built)
            elif i < g + 1 + M:
                rl_train_records.append(built)
            elif i == g + 1 + M:
                rl_val_records.append(built)
            # rest unused
    if insufficient_ids:
        print(f"WARNING: {len(insufficient_ids)} ids have fewer combined rows than needed "
              f"(g + 1 + M + 1). first 5: {insufficient_ids[:5]}")

    seen_k2 = pd.DataFrame(seen_records)
    sft_unique_k2 = pd.DataFrame(sft_unique_records)
    rl_train_k2 = pd.DataFrame(rl_train_records)
    rl_val_k2 = pd.DataFrame(rl_val_records)
    k1_rows = subqas.copy()

    print(f"k=1 rows: {len(k1_rows)}")
    print(f"k=2 SFT seen: {len(seen_k2)}  (uses g per id)")
    print(f"k=2 SFT held-out eval: {len(sft_unique_k2)}  (1 per id)")
    print(f"k=2 RL train: {len(rl_train_k2)}  (M={M} per id)")
    print(f"k=2 RL val pool: {len(rl_val_k2)}  (1 per id)")

    def per_id_first(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby("id", as_index=False).head(1).reset_index(drop=True)

    # ---------- SFT ----------
    sft_train = pd.concat([k1_rows, seen_k2], ignore_index=True).sample(
        frac=1.0, random_state=args.seed
    ).reset_index(drop=True)
    sft_train.to_parquet(SFT_OUT / "train.parquet", index=False)

    per_id_first(k1_rows).to_parquet(SFT_OUT / "train_eval_k1.parquet", index=False)
    per_id_first(seen_k2).to_parquet(SFT_OUT / "train_eval_k2.parquet", index=False)
    sft_unique_k2.to_parquet(SFT_OUT / "train_eval_k2_unique.parquet", index=False)

    origqa_selected = origqa_full[origqa_full["id"].isin(selected_ids)].drop_duplicates("id")
    origqa_selected.to_parquet(SFT_OUT / "train_eval_origqa.parquet", index=False)

    # ---------- RL ----------
    rl_train_k2.to_parquet(RL_OUT / "train.parquet", index=False)

    train_eval = _subsample(rl_train_k2, args.eval_size, args.seed)
    train_eval.to_parquet(RL_OUT / "train_eval.parquet", index=False)

    val = _subsample(rl_val_k2, args.eval_size, args.seed + 1)
    val.to_parquet(RL_OUT / "val.parquet", index=False)

    eval_k1 = _subsample(per_id_first(k1_rows), args.eval_size, args.seed + 2)
    eval_k1.to_parquet(RL_OUT / "eval_k1.parquet", index=False)

    eval_k2_seen = _subsample(per_id_first(seen_k2), args.eval_size, args.seed + 3)
    eval_k2_seen.to_parquet(RL_OUT / "eval_k2_seen.parquet", index=False)

    eval_origqa = _subsample(origqa_selected, args.eval_size, args.seed + 4)
    eval_origqa.to_parquet(RL_OUT / "eval_origqa.parquet", index=False)

    print("\nWrote (sft/):")
    for p in sorted(SFT_OUT.glob("*.parquet")):
        print(f"  {p.name}: {len(pd.read_parquet(p))} rows")
    print("Wrote (rl/):")
    for p in sorted(RL_OUT.glob("*.parquet")):
        print(f"  {p.name}: {len(pd.read_parquet(p))} rows")


if __name__ == "__main__":
    main()
