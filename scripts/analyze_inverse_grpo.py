"""Compare base GRPO vs inverse-mixed GRPO on forward and inverse eval sets."""
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path("/work/hdd/bbsg/twei2/rl/verl/outputs/rl/simpleqa_decompose_grpo/binary_decompose")
MIX_DIR = Path("/work/hdd/bbsg/twei2/rl/verl/outputs/rl/simpleqa_decompose_inverse_grpo/binary_decompose_inverse")

SFT_FWD = BASE_DIR / "base/generations/base_global_step_0__on_eval_val_eval.json"
SFT_INV = BASE_DIR / "base/generations/base_global_step_0__on_eval_val_inverse_qa_eval.json"
RL_BASE_FWD = BASE_DIR / "global_step_760/generations/binary_decompose_global_step_760__on_val_eval.json"
RL_BASE_INV = BASE_DIR / "global_step_760/generations/binary_decompose_global_step_760__on_val_inverse_qa_eval.json"
RL_MIX_COMBINED = MIX_DIR / "global_step_1540/generations/binary_decompose_inverse_global_step_1540__on_val_eval.json"
SFT_MIX_COMBINED = MIX_DIR / "global_step_0/generations/binary_decompose_inverse_global_step_0__on_val_eval.json"


def load(p):
    with open(p) as f:
        return json.load(f)


def is_correct(row):
    g = row.get("graders") or []
    if not g:
        return 0
    return int(g[0].get("evaluation") == "correct")


def is_inverse_row(row):
    aug = row.get("extra_info", {}).get("augmentation")
    return bool(aug) and isinstance(aug, dict) and aug.get("method") == "inverse_qa"


def rows_to_df(rows, label):
    """Return DataFrame indexed by sample id (from extra_info.sample_id) with column `label`."""
    out = []
    for r in rows:
        sid = str(r["extra_info"]["sample_id"])
        q = r["extra_info"].get("question") or r["question"]
        a = r["extra_info"].get("answer") or r["answer"]
        out.append({"id": sid, "question": q, "answer": a, label: is_correct(r)})
    return pd.DataFrame(out)


def split_combined(rows):
    fwd = [r for r in rows if not is_inverse_row(r)]
    inv = [r for r in rows if is_inverse_row(r)]
    return fwd, inv


def acc(rows):
    if not rows:
        return float("nan"), 0
    n = len(rows)
    c = sum(is_correct(r) for r in rows)
    return c / n, n


# Load all
prebase_fwd = load(SFT_FWD)["rows"]    # actually pre-SFT base model
prebase_inv = load(SFT_INV)["rows"]
rl_base_fwd = load(RL_BASE_FWD)["rows"]
rl_base_inv = load(RL_BASE_INV)["rows"]
mix_rows = load(RL_MIX_COMBINED)["rows"]
mix_fwd, mix_inv = split_combined(mix_rows)
sft_mix_rows = load(SFT_MIX_COMBINED)["rows"]
sft_fwd, sft_inv = split_combined(sft_mix_rows)   # SFT init shared by both experiments

# Headline accuracy table
table = [
    ["Pre-SFT base model (base exp 'base/')", *acc(prebase_fwd), *acc(prebase_inv)],
    ["SFT init (mixed exp step 0)", *acc(sft_fwd), *acc(sft_inv)],
    ["Base RL (forward only) step 760", *acc(rl_base_fwd), *acc(rl_base_inv)],
    ["Mixed RL (forward+inverse) step 1540", *acc(mix_fwd), *acc(mix_inv)],
]
headline = pd.DataFrame(
    [(r[0], f"{r[1]:.4f} ({int(r[1]*r[2])}/{r[2]})", f"{r[3]:.4f} ({int(r[3]*r[4])}/{r[4]})") for r in table],
    columns=["Checkpoint", "Forward val acc", "Inverse val acc"],
)
print(headline.to_string(index=False))
print()

# Per-fact dataframe
df_fwd_sft = rows_to_df(sft_fwd, "fwd_sft")
df_fwd_base = rows_to_df(rl_base_fwd, "fwd_rl_base")[["id", "fwd_rl_base"]]
df_fwd_mix = rows_to_df(mix_fwd, "fwd_rl_mixed")[["id", "fwd_rl_mixed"]]
df_inv_sft = rows_to_df(sft_inv, "inv_sft")[["id", "inv_sft"]]
df_inv_base = rows_to_df(rl_base_inv, "inv_rl_base")[["id", "inv_rl_base"]]
df_inv_mix = rows_to_df(mix_inv, "inv_rl_mixed")[["id", "inv_rl_mixed"]]

combo = df_fwd_sft.merge(df_fwd_base, on="id").merge(df_fwd_mix, on="id")
combo = combo.merge(df_inv_sft, on="id").merge(df_inv_base, on="id").merge(df_inv_mix, on="id")
combo = combo[["id", "question", "answer", "fwd_sft", "fwd_rl_base", "fwd_rl_mixed",
               "inv_sft", "inv_rl_base", "inv_rl_mixed"]]
print(f"Per-fact dataframe: {combo.shape}")
print(combo.head())

out_csv = MIX_DIR / "combined_analysis.csv"
combo.to_csv(out_csv, index=False)
print(f"Saved {out_csv}")

# Pattern counts
pat_A = combo[(combo.inv_rl_mixed == 1) & (combo.inv_rl_base == 0) & (combo.fwd_rl_base == 0)]
pat_B = combo[(combo.fwd_rl_base == 1) & (combo.inv_rl_mixed == 1) & (combo.inv_rl_base == 0)]
pat_C = combo[(combo.fwd_rl_base == 0) & (combo.fwd_rl_mixed == 0) &
              (combo.inv_rl_base == 0) & (combo.inv_rl_mixed == 0) & (combo.fwd_sft == 0)]
pat_D = combo[(combo.fwd_rl_base == 1) & (combo.fwd_rl_mixed == 0)]

patterns = {
    "A: inverse training unlocked new fact (inv_rl_mixed=1, inv_rl_base=0, fwd_rl_base=0)": pat_A,
    "B: forward already unlocked, inverse training added inverse pathway (fwd_rl_base=1, inv_rl_mixed=1, inv_rl_base=0)": pat_B,
    "C: permanently stuck (all four RL=0, fwd_sft=0)": pat_C,
    "D: forward regression from mixed training (fwd_rl_base=1, fwd_rl_mixed=0)": pat_D,
}
print()
for k, v in patterns.items():
    print(f"  {k}: {len(v)} / {len(combo)}")

# Save context for markdown
import json as _json
ctx = {
    "headline_rows": table,
    "n_facts": len(combo),
    "pattern_counts": {k.split(":")[0]: len(v) for k, v in patterns.items()},
    "fwd_acc_base": acc(rl_base_fwd)[0],
    "fwd_acc_mix": acc(mix_fwd)[0],
    "inv_acc_base": acc(rl_base_inv)[0],
    "inv_acc_mix": acc(mix_inv)[0],
    "sft_fwd": acc(sft_fwd)[0],
    "sft_inv": acc(sft_inv)[0],
    "examples": {
        "A": pat_A.head(5)[["id", "question", "answer"]].to_dict("records"),
        "B": pat_B.head(5)[["id", "question", "answer"]].to_dict("records"),
        "C": pat_C.head(5)[["id", "question", "answer"]].to_dict("records"),
        "D": pat_D.head(5)[["id", "question", "answer"]].to_dict("records"),
    },
}
with open(MIX_DIR / "_analysis_ctx.json", "w") as f:
    _json.dump(ctx, f, indent=2, default=str)
print("Saved analysis context json.")
