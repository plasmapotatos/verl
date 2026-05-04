"""Compare eval results with vs without richqa prefix priming, on common IDs."""
import argparse
import json
from collections import Counter


def load_rows(path):
    d = json.load(open(path))
    out = {}
    for r in d["rows"]:
        verdict = r["graders"][0]["evaluation"] if r.get("graders") else "missing"
        out[r["id"]] = {
            "verdict": verdict,
            "question": r["question"],
            "answer": r["answer"],
            "predicted": r["graders"][0]["predicted_answer"] if r.get("graders") else None,
        }
    return d.get("metrics", {}), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with_prefix", default="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/richqa/sft_lr1.5e-4_epmax30_seed1/global_step_1290/generations/global_step_1290__on_train_origqa_plus_richqa_prefixes_eval.json")
    ap.add_argument("--no_prefix", default="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/richqa/sft_lr1.5e-4_epmax30_seed1/global_step_1290/generations/sft_lr1.5e-4_epmax30_seed1_global_step_1290__on_train_eval_origqa_eval.json")
    ap.add_argument("--out", default="/work/hdd/bbsg/twei2/rl/verl/outputs/sft/richqa/sft_lr1.5e-4_epmax30_seed1/global_step_1290/generations/ANALYSIS_prefix_vs_noprefix.md")
    args = ap.parse_args()

    m_w, rows_w = load_rows(args.with_prefix)
    m_n, rows_n = load_rows(args.no_prefix)

    common = sorted(set(rows_w) & set(rows_n))
    print(f"with-prefix rows: {len(rows_w)}")
    print(f"no-prefix rows:   {len(rows_n)}")
    print(f"common ids:       {len(common)}")

    def tally(rowdict, ids):
        c = Counter(rowdict[i]["verdict"] for i in ids)
        total = sum(c.values())
        correct = c.get("correct", 0)
        attempted = total - c.get("not_attempted", 0)
        return {
            "total": total,
            "correct": correct,
            "incorrect": c.get("incorrect", 0),
            "not_attempted": c.get("not_attempted", 0),
            "accuracy": correct / total if total else 0.0,
            "accuracy_attempted": correct / attempted if attempted else 0.0,
        }

    s_w = tally(rows_w, common)
    s_n = tally(rows_n, common)

    # transitions
    transitions = Counter()
    flipped_wrong_to_right = []
    flipped_right_to_wrong = []
    for i in common:
        a = rows_n[i]["verdict"]
        b = rows_w[i]["verdict"]
        transitions[(a, b)] += 1
        if a != "correct" and b == "correct":
            flipped_wrong_to_right.append(i)
        if a == "correct" and b != "correct":
            flipped_right_to_wrong.append(i)

    lines = []
    lines.append("# Prefix vs No-Prefix Eval Comparison")
    lines.append("")
    lines.append(f"- with-prefix file: `{args.with_prefix}`")
    lines.append(f"- no-prefix file:   `{args.no_prefix}`")
    lines.append(f"- with-prefix rows: {len(rows_w)}")
    lines.append(f"- no-prefix rows:   {len(rows_n)}")
    lines.append(f"- common ids:       {len(common)}")
    lines.append("")
    lines.append("## Accuracy on common IDs")
    lines.append("")
    lines.append("| condition | correct | incorrect | not_attempted | accuracy | acc_attempted |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| no-prefix  | {s_n['correct']} | {s_n['incorrect']} | {s_n['not_attempted']} | {s_n['accuracy']:.4f} | {s_n['accuracy_attempted']:.4f} |")
    lines.append(f"| +prefix    | {s_w['correct']} | {s_w['incorrect']} | {s_w['not_attempted']} | {s_w['accuracy']:.4f} | {s_w['accuracy_attempted']:.4f} |")
    lines.append(f"| delta      | {s_w['correct']-s_n['correct']:+d} | {s_w['incorrect']-s_n['incorrect']:+d} | {s_w['not_attempted']-s_n['not_attempted']:+d} | {s_w['accuracy']-s_n['accuracy']:+.4f} | {s_w['accuracy_attempted']-s_n['accuracy_attempted']:+.4f} |")
    lines.append("")
    lines.append("## Verdict transitions (no-prefix -> +prefix)")
    lines.append("")
    lines.append("| no-prefix | +prefix | count |")
    lines.append("|---|---|---|")
    for (a, b), c in sorted(transitions.items(), key=lambda x: -x[1]):
        lines.append(f"| {a} | {b} | {c} |")
    lines.append("")
    lines.append(f"- flipped wrong→right: **{len(flipped_wrong_to_right)}**")
    lines.append(f"- flipped right→wrong: **{len(flipped_right_to_wrong)}**")
    lines.append(f"- net gain:            **{len(flipped_wrong_to_right) - len(flipped_right_to_wrong):+d}**")
    lines.append("")

    def sample_block(title, ids, k=8):
        lines.append(f"## Samples: {title}")
        lines.append("")
        for i in ids[:k]:
            lines.append(f"### id {i}")
            lines.append(f"- answer: `{rows_n[i]['answer']}`")
            lines.append(f"- Q (no-prefix): {rows_n[i]['question']}")
            lines.append(f"- Q (+prefix):   {rows_w[i]['question']}")
            lines.append(f"- pred (no-prefix, {rows_n[i]['verdict']}): {rows_n[i]['predicted']}")
            lines.append(f"- pred (+prefix,   {rows_w[i]['verdict']}): {rows_w[i]['predicted']}")
            lines.append("")

    sample_block("wrong -> right (prefix helped)", flipped_wrong_to_right)
    sample_block("right -> wrong (prefix hurt)", flipped_right_to_wrong)

    open(args.out, "w").write("\n".join(lines))
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
