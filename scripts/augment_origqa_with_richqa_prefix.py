"""Augment train_origqa questions by concatenating the first sentence of the
richqa answer prefix (when available).

Reads:
  - train_origqa.parquet (base dataset to copy)
  - richqa_answer_prefixes.parquet (from extract_richqa_answer_prefix.py)

Writes a new parquet where each question becomes:
    "<original question> <first_sentence_of_prefix>"
when a prefix was found; otherwise the row is left unchanged.

Usage:
  python augment_origqa_with_richqa_prefix.py \
    --origqa /path/train_origqa.parquet \
    --prefixes /path/richqa_answer_prefixes.parquet \
    --out /path/train_origqa_plus_richqa_prefix.parquet
"""
import argparse
import re
import pandas as pd


_SENT_END = re.compile(r"([.!?])(\s+|$)")


def first_sentence(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t:
        return None
    m = _SENT_END.search(t)
    if m:
        return t[: m.end(1)].strip()
    # no terminator found — use the whole prefix
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origqa", default="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/sft/train_origqa.parquet")
    ap.add_argument("--prefixes", default="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/sft/richqa_answer_prefixes.parquet")
    ap.add_argument("--out", default="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/sft/train_origqa_plus_richqa_prefix.parquet")
    ap.add_argument("--out_sidecar", default=None,
                    help="Path for a slim sidecar parquet with columns id, question, answer, added_prefix. "
                         "Defaults to <out>.added_prefixes.parquet")
    ap.add_argument("--out_slim", default=None,
                    help="Path for a slim parquet with columns id, question, answer (no added_prefix). "
                         "Defaults to <out>.slim.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.origqa)
    pref = pd.read_parquet(args.prefixes)
    prefix_by_id = dict(zip(pref["id"], pref["prefix"]))

    kept_rows = []
    sidecar_rows = []
    n_total = len(df)
    n_skip = 0
    for _, row in df.iterrows():
        sid = row["id"]
        prefix = prefix_by_id.get(sid)
        sent = first_sentence(prefix) if prefix else None
        if not sent:
            n_skip += 1
            continue

        row = row.copy()
        q = row["question"]
        new_q = f"{q} {sent}"
        sidecar_rows.append({
            "id": sid,
            "question": q,
            "answer": row["answer"],
            "added_prefix": sent,
        })
        row["question"] = new_q

        new_prompt = []
        replaced = False
        for msg in row["prompt"]:
            msg_dict = dict(msg)
            if not replaced and msg_dict.get("role") == "user":
                msg_dict["content"] = new_q
                replaced = True
            new_prompt.append(msg_dict)
        row["prompt"] = new_prompt
        kept_rows.append(row)

    out_df = pd.DataFrame(kept_rows, columns=df.columns).reset_index(drop=True)
    out_df.to_parquet(args.out, index=False)

    sidecar_path = args.out_sidecar or args.out.replace(".parquet", ".added_prefixes.parquet")
    sidecar_df = pd.DataFrame(sidecar_rows, columns=["id", "question", "answer", "added_prefix"])
    sidecar_df.to_parquet(sidecar_path, index=False)

    slim_path = args.out_slim or args.out.replace(".parquet", ".slim.parquet")
    slim_df = sidecar_df[["id", "question", "answer"]].copy()
    slim_df.to_parquet(slim_path, index=False)

    print(f"Input rows: {n_total}")
    print(f"  kept (augmented): {len(out_df)}")
    print(f"  skipped (no prefix): {n_skip}")
    print(f"Saved to: {args.out}")
    print(f"Sidecar saved to: {sidecar_path}")
    print(f"Slim saved to:    {slim_path}")

    for i in range(min(3, len(out_df))):
        print("---")
        print("Q:", out_df.iloc[i]["question"])


if __name__ == "__main__":
    main()
