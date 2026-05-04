"""Build dilution-test SFT datasets.

For each proportion p in {0.25, 0.5, 0.75, 1.0}, sample p * |richqa_ids| ids
(nested: p=0.25 ⊂ 0.5 ⊂ 0.75 ⊂ 1.0), then for each sampled id pick one random
row from the extraneous decompose pool. Concatenate onto the base richqa
train.parquet and write to sft/<p>/train.parquet. Also copy train_eval.parquet
and train_eval_origqa.parquet from sft/0/.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
BASE = Path("/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose_dilution_test/sft")
SOURCE = Path("/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose_and_richqa_capped/sft/train_desimpleqa.parquet")

PROPORTIONS = {
    "0_25": 0.25,
    "0_5": 0.5,
    "0_75": 0.75,
    "1": 1.0,
}

rng = np.random.default_rng(SEED)

base_train = pd.read_parquet(BASE / "0" / "train.parquet")
base_eval = BASE / "0" / "train_eval.parquet"
base_eval_orig = BASE / "0" / "train_eval_origqa.parquet"

extraneous = pd.read_parquet(SOURCE)
extraneous_by_id = {i: g.reset_index(drop=True) for i, g in extraneous.groupby("id")}

richqa_ids = base_train["id"].tolist()
shuffled = richqa_ids.copy()
rng.shuffle(shuffled)

n_total = len(shuffled)
print(f"richqa total ids: {n_total}")

for tag, p in PROPORTIONS.items():
    n = round(p * n_total)
    selected_ids = shuffled[:n]
    out_dir = BASE / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    picked_rows = []
    missing_ids = []
    for rid in selected_ids:
        group = extraneous_by_id.get(rid)
        if group is None or len(group) == 0:
            missing_ids.append(rid)
            continue
        idx = rng.integers(0, len(group))
        picked_rows.append(group.iloc[idx])

    extraneous_df = pd.DataFrame(picked_rows).reset_index(drop=True) if picked_rows else pd.DataFrame(columns=base_train.columns)
    out_train = pd.concat([base_train, extraneous_df], ignore_index=True)
    out_train.to_parquet(out_dir / "train.parquet", index=False)

    shutil.copy2(base_eval, out_dir / "train_eval.parquet")
    shutil.copy2(base_eval_orig, out_dir / "train_eval_origqa.parquet")

    log_path = out_dir / "missing_ids.txt"
    with open(log_path, "w") as f:
        f.write(f"proportion: {p}\n")
        f.write(f"selected_ids: {len(selected_ids)}\n")
        f.write(f"extraneous_added: {len(picked_rows)}\n")
        f.write(f"missing (no match in source): {len(missing_ids)}\n")
        for mid in missing_ids:
            f.write(f"{mid}\n")

    print(f"[{tag}] p={p} selected={len(selected_ids)} added={len(picked_rows)} missing={len(missing_ids)} -> {len(out_train)} rows")

print("done.")
