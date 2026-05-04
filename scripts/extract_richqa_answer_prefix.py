"""Extract prefix of richqa answer before the exact simpleqa answer appears.

For each sample in train_origqa.parquet, find the matching sample (by id) in
train.parquet (rich QA). Locate the first occurrence of the original answer
string within the rich answer, and save everything preceding it as the prefix.

Usage:
  python extract_richqa_answer_prefix.py \
    --origqa /path/train_origqa.parquet \
    --richqa /path/train.parquet \
    --out /path/richqa_answer_prefixes.parquet
"""
import argparse
import pandas as pd


def find_prefix(rich_answer: str, answer: str) -> str | None:
    if not isinstance(rich_answer, str) or not isinstance(answer, str) or not answer:
        return None
    idx = rich_answer.find(answer)
    if idx < 0:
        # try case-insensitive as a fallback
        lower_idx = rich_answer.lower().find(answer.lower())
        if lower_idx < 0:
            return None
        idx = lower_idx
    return rich_answer[:idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origqa", default="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/sft/train_origqa.parquet")
    ap.add_argument("--richqa", default="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/sft/train.parquet")
    ap.add_argument("--out", default="/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/sft/richqa_answer_prefixes.parquet")
    args = ap.parse_args()

    orig = pd.read_parquet(args.origqa)
    rich = pd.read_parquet(args.richqa)

    rich_by_id = {row["id"]: row for _, row in rich.iterrows()}

    rows = []
    n_found = 0
    n_missing_match = 0
    n_missing_rich = 0
    for _, o in orig.iterrows():
        sid = o["id"]
        answer = o["answer"]
        r = rich_by_id.get(sid)
        if r is None:
            n_missing_rich += 1
            rows.append({"id": sid, "answer": answer, "rich_answer": None, "prefix": None, "found": False})
            continue
        rich_answer = r["answer"]
        prefix = find_prefix(rich_answer, answer)
        found = prefix is not None
        if found:
            n_found += 1
        else:
            n_missing_match += 1
        rows.append({
            "id": sid,
            "answer": answer,
            "rich_answer": rich_answer,
            "prefix": prefix,
            "found": found,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(args.out, index=False)
    print(f"Total: {len(out_df)}")
    print(f"  prefix found: {n_found}")
    print(f"  answer not in rich: {n_missing_match}")
    print(f"  no matching rich id: {n_missing_rich}")
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
