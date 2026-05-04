"""Remove SFT rows whose sub-Q/A duplicates the source SimpleQA Q/A.

For each row in the input parquet, compare:
  - sub-Q/A: row['question'], row['answer']
  - simpleqa Q/A: row['extra_info']['original_question'], row['extra_info']['original_answer']

A row is dropped when an LLM judges that the two Q/A pairs are essentially
asking for the same fact with the same answer. The LLM sees both Q+A pairs at
once (so semantic context matters) and decides in a single call.

Usage:
  python scripts/desimpleqa_parquet.py <input.parquet> [-o out.parquet]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verl.augment.openai_client import OpenAIClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are judging whether two factual question/answer pairs are duplicates.

You will be given numbered pairs. Each pair contains:
  - Q1/A1: the original SimpleQA question and gold answer
  - Q2/A2: a sub-question and its answer derived from the same source

Return DUPLICATE when BOTH conditions hold:
  (a) The answers refer to the same underlying fact/entity (identical, or
      trivially reformatted — "October 6, 2020" vs "Oct 6 2020" vs "10/6/2020";
      "Michio Sugeno" vs "Michio Sugeno."; "January 2008" vs "In January 2008.").
      Minor additions like full dates vs partial dates DO still count as the
      same answer if the shorter one is a prefix/subset of the longer one.
  (b) The questions, read together with their answers, ask for the same fact.
      Phrasing, word order, abbreviation, or added scope qualifiers do not
      matter if a correct responder would give the same answer.

Return DIFFERENT otherwise. In particular return DIFFERENT when:
  - Answers share a string but denote different things (same date, different
    events; same person name referring to different roles).
  - Questions look similar but probe different facts that happen to share a
    surface answer token.
  - Answers disagree or only partially overlap in meaning (e.g. "October 2020"
    vs "October 6, 2020, in Dallas" is still DUPLICATE — subset/prefix is fine;
    but "October 2020" vs "October 2021" is DIFFERENT).

### Examples

[ex1] Q1: Who received the IEEE Frank Rosenblatt Award in 2010?   A1: Michio Sugeno
      Q2: Who received the IEEE Frank Rosenblatt Award in 2010?   A2: Michio Sugeno.
      -> DUPLICATE (identical question, answer differs only by punctuation)

[ex2] Q1: What were the month and year when Obama told Christianity Today, "I am a Christian..."?   A1: January 2008
      Q2: When did Obama state that he is a devout Christian?   A2: In January 2008.
      -> DUPLICATE (same underlying fact, same answer)

[ex3] Q1: What month, day, and year did Brenda Gayle Hayes die?   A1: October 6, 2020
      Q2: What was the date of death of W. V. Grant's spouse?   A2: October 6, 2020
      -> DUPLICATE (W. V. Grant's spouse is Brenda Gayle Hayes; same fact, same answer)

[ex4] Q1: On what date did Ben Eckerson complete his snow-globe record?   A1: December 15, 2007
      Q2: Who set the world record for the longest time spent in a snow globe in 2007?   A2: Ben Eckerson
      -> DIFFERENT (one asks for a date, the other for a person; different facts)

[ex5] Q1: In what year was the Battle of Hastings?   A1: 1066
      Q2: In what year did William the Conqueror invade England?   A2: 1066
      -> DUPLICATE (same event, same answer; topic link is tight)

[ex6] Q1: When did World War II end in Europe?   A1: May 1945
      Q2: When did World War II end in the Pacific?   A2: September 1945
      -> DIFFERENT (related topic, different fact, different answer)

[ex7] Q1: Who directed Inception?   A1: Christopher Nolan
      Q2: Who directed Interstellar?   A2: Christopher Nolan
      -> DIFFERENT (answer string matches but the questions ask about different films)

[ex8] Q1: What is the capital of France?   A1: Paris
      Q2: Which city hosts the French National Assembly?   A2: Paris, France
      -> DUPLICATE (same city; short answer is a subset of the longer one; same fact about the French capital)

Output format: a JSON array of objects, one per input pair, in any order:
  [{"i": 1, "verdict": "DUPLICATE"}, {"i": 2, "verdict": "DIFFERENT"}, ...]
Output only the JSON array, nothing else."""


def judge_batch(client: OpenAIClient, pairs: list[tuple[int, str, str, str, str]]) -> dict[int, str]:
    """pairs: list of (idx, q1, a1, q2, a2). Returns idx -> 'DUPLICATE' | 'DIFFERENT'."""
    lines = []
    for idx, q1, a1, q2, a2 in pairs:
        lines.append(
            f"[{idx}]\n"
            f"  Q1: {q1}\n  A1: {a1}\n"
            f"  Q2: {q2}\n  A2: {a2}"
        )
    user_prompt = "Judge these pairs:\n\n" + "\n\n".join(lines)

    raw = client.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=4096,
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    out: dict[int, str] = {}
    try:
        parsed = json.loads(raw)
        for entry in parsed:
            out[int(entry["i"])] = str(entry["verdict"]).upper()
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.warning("Failed to parse LLM response; defaulting batch to DIFFERENT (keep rows).")
        for idx, *_ in pairs:
            out[idx] = "DIFFERENT"
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input parquet path")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Filtered output parquet (default: <input>_desimpleqa.parquet)")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Q/A pairs per LLM call (default 10)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N rows (for debugging)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the batches that would be sent and exit")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    if args.output is None:
        args.output = args.input.with_name(args.input.stem + "_desimpleqa.parquet")

    df = pd.read_parquet(args.input)
    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {args.input}")

    sub_qs = df["question"].tolist()
    sub_as = df["answer"].tolist()
    orig_qs, orig_as = [], []
    for ei in df["extra_info"].tolist():
        orig_qs.append(ei.get("original_question", "") if ei is not None else "")
        orig_as.append(ei.get("original_answer", "") if ei is not None else "")

    all_idxs = list(range(len(df)))
    batches: list[list[tuple[int, str, str, str, str]]] = []
    for i in range(0, len(all_idxs), args.batch_size):
        chunk = all_idxs[i : i + args.batch_size]
        batches.append([(idx, orig_qs[idx], orig_as[idx], sub_qs[idx], sub_as[idx]) for idx in chunk])
    print(f"Prepared {len(batches)} batches (batch_size={args.batch_size}, workers={args.num_workers})")

    if args.dry_run:
        print("\n--- First batch preview ---")
        for tup in batches[0][:5]:
            idx, q1, a1, q2, a2 = tup
            print(f"[{idx}]\n  Q1: {q1}\n  A1: {a1}\n  Q2: {q2}\n  A2: {a2}\n")
        return

    client = OpenAIClient(model=args.model)

    verdicts: dict[int, str] = {}
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(judge_batch, client, b) for b in batches]
        for fut in concurrent.futures.as_completed(futures):
            verdicts.update(fut.result())
            completed += 1
            if completed % 10 == 0 or completed == len(batches):
                print(f"  {completed}/{len(batches)} batches done")

    drop_idxs = {i for i in all_idxs if verdicts.get(i) == "DUPLICATE"}
    print(f"LLM dropped: {len(drop_idxs)} / {len(df)} rows")

    keep_mask = [i not in drop_idxs for i in range(len(df))]
    kept = df[keep_mask].reset_index(drop=True)
    dropped = df[[not k for k in keep_mask]].reset_index(drop=True)

    kept.to_parquet(args.output)
    print(f"Wrote {len(kept)} rows -> {args.output}")

    dropped_path = args.output.with_name(args.output.stem + "_dropped.parquet")
    dropped.to_parquet(dropped_path)
    print(f"Wrote {len(dropped)} dropped rows -> {dropped_path}")

    # Also emit a human-readable audit log.
    audit_path = args.output.with_name(args.output.stem + "_dropped.txt")
    with audit_path.open("w") as f:
        for _, r in dropped.iterrows():
            ei = r["extra_info"]
            f.write(f"id={r['id']}  orig_sample_id={ei.get('original_sample_id', ei.get('augmentation', {}).get('original_sample_id'))}\n")
            f.write(f"  SIMPLEQA Q: {ei.get('original_question')}\n")
            f.write(f"  SIMPLEQA A: {ei.get('original_answer')}\n")
            f.write(f"  SUB     Q: {r['question']}\n")
            f.write(f"  SUB     A: {r['answer']}\n\n")
    print(f"Wrote audit log -> {audit_path}")


if __name__ == "__main__":
    main()
