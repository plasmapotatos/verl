#!/usr/bin/env python3
"""Re-grade the refusal_unknown_test JSONL outputs using the SimpleQA LLM judge.

The original `compare_refusal_on_unknown.py` flagged refusals via a keyword
regex (`_is_not_attempted`), which dramatically under-counts hedged refusals
("I'm sorry, I don't have specific information ..."). This script reuses the
project's OpenAIJudge + simpleqa_judge.txt to relabel each response as
correct / incorrect / not_attempted, and writes a new summary alongside.

Usage (inside the apptainer container):
  apptainer exec /work/hdd/bbsg/twei2/rl/torch2501.sif python \
      scripts/regrade_refusal_unknown.py \
      --in-dir outputs/analysis/refusal_unknown_test \
      --workers 16
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from verl.eval.judge.openai_judge import OpenAIJudge  # noqa: E402

PROMPT_PATH = REPO_ROOT / "verl" / "eval" / "judge" / "prompts" / "simpleqa_judge.txt"


def load_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_prompt(template: str, question: str, target: str, predicted: str) -> str:
    return template.format(question=question, target=target, predicted_answer=predicted)


def parse_judge_output(text: str) -> str:
    text = (text or "").strip().upper()
    if "NOT_ATTEMPTED" in text:
        return "not_attempted"
    if "CORRECT" in text or " A " in f" {text} ":
        return "correct"
    if " C " in f" {text} ":
        return "not_attempted"
    return "incorrect"


def grade_one(judge: OpenAIJudge, template: str, row: dict) -> dict:
    prompt = build_prompt(template, row["question"], row["gold_answer"], row["response"])
    raw = judge.generate(prompt)
    label = parse_judge_output(raw)
    out = dict(row)
    out["judge_raw"] = raw
    out["judge_label"] = label
    return out


def regrade_file(in_path: Path, out_path: Path, judge: OpenAIJudge, template: str, workers: int) -> dict:
    with in_path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"[{in_path.name}] grading {len(rows)} rows with {workers} workers", flush=True)

    results: list[dict] = [None] * len(rows)  # type: ignore
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(grade_one, judge, template, r): i for i, r in enumerate(rows)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                r = dict(rows[i])
                r["judge_raw"] = f"<error: {e}>"
                r["judge_label"] = "error"
                results[i] = r
            done += 1
            if done % 50 == 0 or done == len(rows):
                print(f"  [{in_path.name}] {done}/{len(rows)}", flush=True)

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = {"correct": 0, "incorrect": 0, "not_attempted": 0, "error": 0}
    regex_refusals = 0
    for r in results:
        counts[r.get("judge_label", "error")] = counts.get(r.get("judge_label", "error"), 0) + 1
        if r.get("is_refusal"):
            regex_refusals += 1
    n = len(results)
    summary = {
        "n": n,
        "judge_correct": counts["correct"],
        "judge_incorrect": counts["incorrect"],
        "judge_not_attempted": counts["not_attempted"],
        "judge_error": counts["error"],
        "judge_refusal_rate": counts["not_attempted"] / n if n else 0.0,
        "judge_accuracy": counts["correct"] / n if n else 0.0,
        "judge_attempted": (counts["correct"] + counts["incorrect"]),
        "judge_attempt_rate": (counts["correct"] + counts["incorrect"]) / n if n else 0.0,
        "regex_refusals": regex_refusals,
        "regex_refusal_rate": regex_refusals / n if n else 0.0,
    }
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="/work/hdd/bbsg/twei2/rl/verl/outputs/analysis/refusal_unknown_test")
    p.add_argument("--out-dir", default=None, help="Defaults to <in-dir>/regraded")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--files", nargs="*", default=None,
                   help="Specific .jsonl filenames inside --in-dir; defaults to all *.jsonl")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (in_dir / "regraded")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.jsonl")) if not args.files else [in_dir / f for f in args.files]
    if not files:
        raise SystemExit(f"no jsonl files in {in_dir}")

    template = load_template()
    judge = OpenAIJudge(model=args.model)

    overall = {}
    for fp in files:
        out_fp = out_dir / fp.name
        summary = regrade_file(fp, out_fp, judge, template, args.workers)
        overall[fp.stem] = summary
        print(f"  -> {out_fp}  refusal={summary['judge_refusal_rate']:.3f} "
              f"acc={summary['judge_accuracy']:.3f} (regex_refusal={summary['regex_refusal_rate']:.3f})",
              flush=True)

    summary_path = out_dir / "summary_judge.json"
    summary_path.write_text(json.dumps(overall, indent=2))

    print("\n=== Refusal & accuracy under LLM judge (gpt-4o-mini) ===")
    print(f"{'model':<32s}  {'n':>4s}  {'C':>4s}  {'I':>4s}  {'NA':>4s}  {'judge_refusal':>13s}  {'regex_refusal':>13s}")
    for name, s in overall.items():
        print(f"{name:<32s}  {s['n']:>4d}  {s['judge_correct']:>4d}  {s['judge_incorrect']:>4d}  "
              f"{s['judge_not_attempted']:>4d}  {s['judge_refusal_rate']:>13.3f}  {s['regex_refusal_rate']:>13.3f}")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
