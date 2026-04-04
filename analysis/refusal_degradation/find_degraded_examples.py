"""
Find specific examples that go from very-high-confidence correct at step 0
(e.g. ≥26/32 = 81% correct) to mostly-IDK at step 480.

Outputs:
  - fig_degraded_examples.png  : question+trajectory plots with sample responses
  - degraded_examples.txt      : full text dump of examples with actual model outputs
"""

import json
import os
import textwrap
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

BASE_ROOT = "/work/hdd/bbsg/twei2/rl/verl/outputs/rl/simpleqa_rich_sft_grpo_refusal"
EXPS = {
    "binary":           f"{BASE_ROOT}/simpleqa_rich_sft_grpo_refusal_binary",
    "ternary_adaptive": f"{BASE_ROOT}/simpleqa_rich_sft_grpo_refusal_ternary_adaptive",
}
STEPS = [0, 100, 200, 300, 400, 480]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Thresholds for selecting examples
CORRECT_FRAC_AT_0_MIN   = 0.80   # ≥80% correct at step 0
IDK_FRAC_AT_480_MIN     = 0.70   # ≥70% IDK at step 480


def load_eval(exp_base, step, set_name):
    path = f"{exp_base}/global_step_{step}/pass@k/{set_name}/eval.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def row_summary(row):
    evals = [g["evaluation"] for g in row["graders"]]
    return {
        "id": row["id"],
        "question": row["question"],
        "answer": row.get("answer", ""),
        "correct": evals.count("correct"),
        "incorrect": evals.count("incorrect"),
        "not_attempted": evals.count("not_attempted"),
        "total": len(evals),
        "responses": row.get("responses", []),
        "predicted_answers": [g["predicted_answer"] for g in row["graders"]],
        "evaluations": evals,
    }


def find_degraded_examples(exp_base, set_name,
                            correct_thresh=CORRECT_FRAC_AT_0_MIN,
                            idk_thresh=IDK_FRAC_AT_480_MIN):
    """
    Returns list of dicts, one per degraded question:
      {id, question, answer, traj, step0_summary, step480_summary}
    Sorted by step-0 correct fraction descending (highest confidence first).
    """
    d0 = load_eval(exp_base, 0, set_name)
    d480 = load_eval(exp_base, 480, set_name)
    if d0 is None or d480 is None:
        return []

    step0_by_id   = {r["id"]: row_summary(r) for r in d0["rows"]}
    step480_by_id = {r["id"]: row_summary(r) for r in d480["rows"]}

    # Find IDs that were high-correct at 0 and high-IDK at 480
    candidates = []
    for sid, s0 in step0_by_id.items():
        if sid not in step480_by_id:
            continue
        s480 = step480_by_id[sid]
        frac_correct_0   = s0["correct"]  / s0["total"]
        frac_idk_480     = s480["not_attempted"] / s480["total"]
        if frac_correct_0 >= correct_thresh and frac_idk_480 >= idk_thresh:
            candidates.append({
                "id": sid,
                "question": s0["question"],
                "answer": s0["answer"],
                "frac_correct_0": frac_correct_0,
                "frac_idk_480": frac_idk_480,
                "step0": s0,
                "step480": s480,
            })

    # Sort by step-0 correct fraction desc, then IDK-at-480 desc
    candidates.sort(key=lambda x: (-x["frac_correct_0"], -x["frac_idk_480"]))

    # Build full trajectory for each candidate
    all_data = {}
    for step in STEPS:
        d = load_eval(exp_base, step, set_name)
        if d is None:
            continue
        for row in d["rows"]:
            sid = row["id"]
            if sid not in all_data:
                all_data[sid] = {}
            all_data[sid][step] = row_summary(row)

    for c in candidates:
        c["traj"] = all_data.get(c["id"], {})

    return candidates


def print_examples(examples, exp_name, set_name, outfile, n=20):
    """Write detailed text dump of top-N degraded examples."""
    outfile.write(f"\n{'='*80}\n")
    outfile.write(f"EXPERIMENT: {exp_name}   SET: {set_name}\n")
    outfile.write(f"{'='*80}\n")
    outfile.write(f"Found {len(examples)} examples (≥{CORRECT_FRAC_AT_0_MIN*100:.0f}% correct at step 0 → "
                  f"≥{IDK_FRAC_AT_480_MIN*100:.0f}% IDK at step 480)\n\n")

    for rank, ex in enumerate(examples[:n], 1):
        outfile.write(f"--- Example {rank} (id={ex['id']}) ---\n")
        outfile.write(f"QUESTION : {ex['question']}\n")
        outfile.write(f"GT ANSWER: {ex['answer']}\n")
        s0, s480 = ex["step0"], ex["step480"]
        outfile.write(f"Step 0  : {s0['correct']}/{s0['total']} correct, "
                      f"{s0['not_attempted']}/{s0['total']} IDK, "
                      f"{s0['incorrect']}/{s0['total']} incorrect\n")
        outfile.write(f"Step 480: {s480['correct']}/{s480['total']} correct, "
                      f"{s480['not_attempted']}/{s480['total']} IDK, "
                      f"{s480['incorrect']}/{s480['total']} incorrect\n")

        # Trajectory summary
        traj_str = "  Trajectory: "
        for step in STEPS:
            if step in ex["traj"]:
                t = ex["traj"][step]
                traj_str += f"s{step}({t['correct']}C/{t['not_attempted']}I) "
        outfile.write(traj_str + "\n")

        # Sample responses at step 0 (show a correct one)
        correct_responses_0 = [
            (ex["traj"][0]["responses"][i], ex["traj"][0]["predicted_answers"][i])
            for i in range(len(ex["traj"].get(0, {}).get("responses", [])))
            if ex["traj"][0]["evaluations"][i] == "correct"
        ]
        idk_responses_480 = [
            (ex["traj"][480]["responses"][i], ex["traj"][480]["predicted_answers"][i])
            for i in range(len(ex["traj"].get(480, {}).get("responses", [])))
            if ex["traj"][480]["evaluations"][i] == "not_attempted"
        ]

        if correct_responses_0:
            resp, pred = correct_responses_0[0]
            outfile.write(f"\n  [Step 0 - CORRECT sample]\n")
            outfile.write(f"  Predicted: {pred}\n")
            wrapped = textwrap.fill(resp, width=90, initial_indent="    ", subsequent_indent="    ")
            outfile.write(wrapped[:800] + ("\n    ...[truncated]" if len(resp) > 800 else "") + "\n")

        if idk_responses_480:
            resp, pred = idk_responses_480[0]
            outfile.write(f"\n  [Step 480 - IDK sample]\n")
            outfile.write(f"  Predicted: {pred}\n")
            wrapped = textwrap.fill(resp, width=90, initial_indent="    ", subsequent_indent="    ")
            outfile.write(wrapped[:800] + ("\n    ...[truncated]" if len(resp) > 800 else "") + "\n")

        outfile.write("\n")


def fig_degraded_examples(all_examples_by_exp, n_show=8):
    """
    For each experiment, plot the top-N degraded examples as stacked bar trajectories
    with the question text as title.
    """
    print("Plotting figure: degraded examples …")

    for exp_name, examples in all_examples_by_exp.items():
        if not examples:
            print(f"  {exp_name}: no examples found")
            continue

        n = min(n_show, len(examples))
        fig, axes = plt.subplots(2, 4, figsize=(20, 9)) if n > 4 else plt.subplots(1, n, figsize=(5*n, 5))
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        fig.suptitle(
            f"{exp_name} – VAL set: top-{n} examples degrading from high-confidence correct → IDK\n"
            f"(sorted by step-0 correct fraction desc; stacked = correct/IDK/incorrect fraction of 32 samples)",
            fontsize=11,
        )

        for idx, ex in enumerate(examples[:n]):
            ax = axes_flat[idx]
            traj = ex["traj"]
            steps_avail = sorted(traj.keys())
            frac_c = [traj[s]["correct"]       / traj[s]["total"] for s in steps_avail]
            frac_n = [traj[s]["not_attempted"] / traj[s]["total"] for s in steps_avail]
            frac_i = [traj[s]["incorrect"]     / traj[s]["total"] for s in steps_avail]

            xs = list(range(len(steps_avail)))
            ax.stackplot(xs, frac_c, frac_n, frac_i,
                         labels=["correct", "IDK", "incorrect"],
                         colors=["#4CAF50", "#FF9800", "#F44336"], alpha=0.85)

            ax.set_xticks(xs)
            ax.set_xticklabels([f"s{s}" for s in steps_avail], fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.tick_params(labelsize=7)

            q = ex["question"]
            q_wrapped = "\n".join(textwrap.wrap(q, width=42))
            s0 = ex["step0"]
            s480 = ex["step480"]
            title = (
                f"#{idx+1} id={ex['id']}\n{q_wrapped}\n"
                f"GT: {str(ex['answer'])[:40]}\n"
                f"s0: {s0['correct']}/{s0['total']}C  →  s480: {s480['not_attempted']}/{s480['total']}IDK"
            )
            ax.set_title(title, fontsize=6.5, loc="left")
            if idx == 0:
                ax.set_ylabel("fraction of 32 responses")
                ax.legend(fontsize=7, loc="lower left")

        # Hide unused subplots
        for idx in range(n, len(axes_flat)):
            axes_flat[idx].axis("off")

        plt.tight_layout()
        path = f"{OUT_DIR}/fig_degraded_examples_{exp_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved → {path}")


def main():
    SET_NAME = "train_richqa_answer_origqa_frac0.2"  # val set

    all_examples = {}
    txt_path = f"{OUT_DIR}/degraded_examples.txt"
    with open(txt_path, "w") as outfile:
        outfile.write("DEGRADED EXAMPLES: high-confidence correct at step 0 → IDK at step 480\n")
        outfile.write(f"Threshold: ≥{CORRECT_FRAC_AT_0_MIN*100:.0f}% correct at step 0, "
                      f"≥{IDK_FRAC_AT_480_MIN*100:.0f}% IDK at step 480\n")

        for exp_name, exp_base in EXPS.items():
            print(f"\nProcessing {exp_name} …")
            examples = find_degraded_examples(exp_base, SET_NAME)
            all_examples[exp_name] = examples
            print(f"  Found {len(examples)} degraded examples")
            if examples:
                print(f"  Top 5 by step-0 correct fraction:")
                for ex in examples[:5]:
                    print(f"    id={ex['id']}  s0={ex['frac_correct_0']:.2f}  s480_idk={ex['frac_idk_480']:.2f}  Q: {ex['question'][:70]}")
            print_examples(examples, exp_name, SET_NAME, outfile, n=20)

    print(f"\nText dump → {txt_path}")
    fig_degraded_examples(all_examples, n_show=8)
    print("Done.")


if __name__ == "__main__":
    main()
