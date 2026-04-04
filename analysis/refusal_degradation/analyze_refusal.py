"""
Analysis: Why does GRPO refusal training cause correct val answers to become IDK?

Three reward experiments:
  - binary:            outputs/rl/.../simpleqa_rich_sft_grpo_refusal_binary
  - ternary_adaptive:  outputs/rl/.../simpleqa_rich_sft_grpo_refusal_ternary_adaptive
  - ternary_static:    outputs/rl/.../simpleqa_rich_sft_grpo_refusal_ternary_static

Pass@k evaluation sets (inside each global_step_X/pass@k/):
  GRPO train set  - train_richqa_answer_origqa_frac0.8_frac0.2  (questions trained on during GRPO)
  GRPO val set    - train_richqa_answer_origqa_frac0.2           (questions seen only during SFT)
  (analogous refusal sets also exist)

Observation: on the val set (but not train), originally-correct answers degrade into IDK/not_attempted.
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_ROOT = "/work/hdd/bbsg/twei2/rl/verl/outputs/rl/simpleqa_rich_sft_grpo_refusal"
EXPS = {
    "binary":           f"{BASE_ROOT}/simpleqa_rich_sft_grpo_refusal_binary",
    "ternary_adaptive": f"{BASE_ROOT}/simpleqa_rich_sft_grpo_refusal_ternary_adaptive",
    # ternary_static has no val pass@k, only train frac0.8 from step 100 onward
    "ternary_static":   f"{BASE_ROOT}/simpleqa_rich_sft_grpo_refusal_ternary_static",
}
STEPS = [0, 100, 200, 300, 400, 480]
SETS = {
    "val_answer":   "train_richqa_answer_origqa_frac0.2",
    "train_answer": "train_richqa_answer_origqa_frac0.8_frac0.2",
    "val_refusal":  "train_richqa_refusal_origqa_frac0.2",
    "train_refusal":"train_richqa_refusal_origqa_frac0.8_frac0.2",
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_eval(exp_base, step, set_key):
    path = f"{exp_base}/global_step_{step}/pass@k/{SETS[set_key]}/eval.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_passatk(exp_base, step, set_key):
    path = f"{exp_base}/global_step_{step}/pass@k/{SETS[set_key]}/pass_at_k.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def row_summary(row):
    """Return per-question counts of correct/incorrect/not_attempted across k=32 samples."""
    evals = [g["evaluation"] for g in row["graders"]]
    return {
        "id": row["id"],
        "question": row["question"],
        "correct": evals.count("correct"),
        "incorrect": evals.count("incorrect"),
        "not_attempted": evals.count("not_attempted"),
        "total": len(evals),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Aggregate metrics over training steps (all exps, val vs train)
# ══════════════════════════════════════════════════════════════════════════════

def fig1_aggregate_metrics():
    print("Figure 1: aggregate metrics over steps …")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    fig.suptitle(
        "Aggregate response-type counts & pass@k across GRPO training steps\n"
        "(VAL = held-out SFT facts, TRAIN = facts seen during GRPO)",
        fontsize=13,
    )

    exp_names = ["binary", "ternary_adaptive"]  # ternary_static lacks val
    set_pairs = [("val_answer", "train_answer"), ("val_answer", "train_answer")]

    for col, exp in enumerate(exp_names):
        exp_base = EXPS[exp]
        for row_idx, split in enumerate(["val_answer", "train_answer"]):
            ax = axes[row_idx][col]
            steps_found, n_correct, n_idk, n_inc, passatk_vals = [], [], [], [], []
            for step in STEPS:
                d = load_passatk(exp_base, step, split)
                if d is None:
                    continue
                rm = d["response_metrics"]
                pak = d["pass_at_k"]
                steps_found.append(step)
                n_correct.append(rm["correct"])
                n_idk.append(rm["not_attempted"])
                n_inc.append(rm["incorrect"])
                passatk_vals.append(pak["pass_at_k"])

            ax2 = ax.twinx()
            ax.bar(steps_found, n_correct, label="correct",       color="#4CAF50", alpha=0.7, width=40)
            ax.bar(steps_found, n_idk,     label="not_attempted",  color="#FF9800", alpha=0.7, width=40,
                   bottom=n_correct)
            ax.bar(steps_found, n_inc,     label="incorrect",      color="#F44336", alpha=0.7, width=40,
                   bottom=[c+i for c,i in zip(n_correct, n_idk)])
            ax2.plot(steps_found, passatk_vals, "k--o", label="pass@k", linewidth=2, markersize=5)
            ax2.set_ylim(0, 0.8)
            ax2.set_ylabel("pass@k", fontsize=9)

            split_label = "VAL (held-out)" if "val" in split else "TRAIN (GRPO)"
            ax.set_title(f"{exp}\n{split_label}", fontsize=10)
            ax.set_xlabel("global step")
            ax.set_ylabel("# responses (k=32)")
            if row_idx == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=7)

    # ternary_static — only train_answer available, from step 100+
    for row_idx, split in enumerate(["train_answer"]):
        ax = axes[row_idx][2]
        exp_base = EXPS["ternary_static"]
        steps_found, n_correct, n_idk, n_inc, passatk_vals = [], [], [], [], []
        for step in STEPS:
            d = load_passatk(exp_base, step, split)
            if d is None:
                continue
            rm = d["response_metrics"]
            pak = d["pass_at_k"]
            steps_found.append(step)
            n_correct.append(rm["correct"])
            n_idk.append(rm["not_attempted"])
            n_inc.append(rm["incorrect"])
            passatk_vals.append(pak["pass_at_k"])
        ax2 = ax.twinx()
        ax.bar(steps_found, n_correct, label="correct",       color="#4CAF50", alpha=0.7, width=40)
        ax.bar(steps_found, n_idk,     label="not_attempted",  color="#FF9800", alpha=0.7, width=40,
               bottom=n_correct)
        ax.bar(steps_found, n_inc,     label="incorrect",      color="#F44336", alpha=0.7, width=40,
               bottom=[c+i for c,i in zip(n_correct, n_idk)])
        ax2.plot(steps_found, passatk_vals, "k--o", label="pass@k", linewidth=2, markersize=5)
        ax2.set_ylim(0, 0.8)
        ax2.set_ylabel("pass@k", fontsize=9)
        ax.set_title("ternary_static\nTRAIN only (no val eval)", fontsize=10)
        ax.set_xlabel("global step")
        ax.set_ylabel("# responses (k=32)")

    axes[1][2].axis("off")  # no data for ternary_static val
    axes[1][2].text(0.5, 0.5, "ternary_static: no val eval available",
                    ha="center", va="center", transform=axes[1][2].transAxes, fontsize=10, color="gray")

    plt.tight_layout()
    path = f"{OUT_DIR}/fig1_aggregate_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Per-sample fate of step-0 CORRECT questions across steps
# Key question: do step-0 correct answers become IDK, or something else?
# ══════════════════════════════════════════════════════════════════════════════

def build_per_sample_trajectory(exp_base, split, steps=STEPS):
    """
    Returns a dict: {sample_id: {step: {'correct':int,'incorrect':int,'not_attempted':int}}}
    Only for samples with at least 1 correct response at step 0.
    """
    # Load step-0 to find which IDs were correct
    d0 = load_eval(exp_base, steps[0], split)
    if d0 is None:
        return None, None
    step0_rows = {r["id"]: row_summary(r) for r in d0["rows"]}
    correct_at_0 = {sid for sid, s in step0_rows.items() if s["correct"] > 0}
    all_correct_at_0 = {sid for sid, s in step0_rows.items() if s["correct"] == s["total"]}

    trajectories = defaultdict(dict)
    for step in steps:
        d = load_eval(exp_base, step, split)
        if d is None:
            continue
        for row in d["rows"]:
            sid = row["id"]
            if sid in correct_at_0:
                trajectories[sid][step] = row_summary(row)

    return trajectories, {"correct_at_0": correct_at_0, "all_correct_at_0": all_correct_at_0,
                          "step0_rows": step0_rows}


def fig2_persample_fate():
    print("Figure 2: per-sample fate of step-0 correct questions …")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Fate of questions that had ≥1 correct answer at step 0\n"
        "Each line = one question. Y-axis = fraction of 32 samples that are 'correct'",
        fontsize=12,
    )

    panels = [
        ("binary",           "val_answer",   axes[0][0], "binary – VAL"),
        ("binary",           "train_answer", axes[0][1], "binary – TRAIN"),
        ("ternary_adaptive", "val_answer",   axes[1][0], "ternary_adaptive – VAL"),
        ("ternary_adaptive", "train_answer", axes[1][1], "ternary_adaptive – TRAIN"),
    ]

    for exp, split, ax, title in panels:
        exp_base = EXPS[exp]
        trajs, meta = build_per_sample_trajectory(exp_base, split)
        if trajs is None:
            ax.text(0.5, 0.5, "data unavailable", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        step0_rows = meta["step0_rows"]
        n_correct_at_0 = len(meta["correct_at_0"])

        # Categorize by step-0 correct fraction: "all correct" vs "mixed"
        colors_all  = "#1565C0"  # deep blue – was all-correct at step 0
        colors_some = "#90CAF9"  # light blue – had some correct at step 0

        n_stayed = 0
        n_degraded = 0
        n_total = 0

        for sid, traj in trajs.items():
            xs = sorted(traj.keys())
            ys = [traj[s]["correct"] / traj[s]["total"] for s in xs]
            is_all = step0_rows[sid]["correct"] == step0_rows[sid]["total"]
            color = colors_all if is_all else colors_some
            ax.plot(xs, ys, color=color, alpha=0.3, linewidth=0.8)

            # Count degradation: step-0 fraction > 0, step-480 fraction == 0
            if xs[-1] == 480:
                n_total += 1
                if traj[xs[-1]]["correct"] == 0:
                    n_degraded += 1
                else:
                    n_stayed += 1

        # Overlay mean trajectory for "all correct at 0" group
        all_correct_ids = meta["all_correct_at_0"]
        if all_correct_ids:
            mean_ys = []
            for step in STEPS:
                vals = [trajs[sid][step]["correct"] / trajs[sid][step]["total"]
                        for sid in all_correct_ids if step in trajs[sid]]
                mean_ys.append(np.mean(vals) if vals else np.nan)
            ax.plot(STEPS, mean_ys, color="#B71C1C", linewidth=2.5, linestyle="--",
                    label=f"mean of all-correct-at-0 (n={len(all_correct_ids)})")

        ax.set_title(f"{title}\n{n_degraded}/{n_total} originally-correct Qs fully degraded to IDK by step 480",
                     fontsize=10)
        ax.set_xlabel("global step")
        ax.set_ylabel("fraction correct (out of 32)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(STEPS)

        patch_all  = mpatches.Patch(color=colors_all,  label=f"all-correct at step 0 (n={len(all_correct_ids)})")
        patch_some = mpatches.Patch(color=colors_some, label=f"some-correct at step 0 (n={n_correct_at_0- len(all_correct_ids)})")
        ax.legend(handles=[patch_all, patch_some] + ax.get_lines()[-1:], fontsize=8)

    plt.tight_layout()
    path = f"{OUT_DIR}/fig2_persample_fate.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Where do the "correct" responses go? Stack plot per question group
# Groups: (a) all-correct at 0, (b) mixed at 0, (c) all-idk at 0
# ══════════════════════════════════════════════════════════════════════════════

def fig3_fate_stack():
    print("Figure 3: stacked fate by step-0 group …")

    group_names = ["all-correct", "mixed", "all-IDK/inc"]
    exp_list = ["binary", "ternary_adaptive"]
    fig, axes = plt.subplots(len(group_names), len(exp_list), figsize=(12, 10))
    fig.suptitle(
        "Average response distribution (correct / IDK / incorrect)\n"
        "grouped by step-0 behavior  –  VAL set only",
        fontsize=12,
    )

    for col, exp in enumerate(exp_list):
        exp_base = EXPS[exp]
        trajs, meta = build_per_sample_trajectory(exp_base, "val_answer")
        step0_rows = meta["step0_rows"]

        # Group questions by step-0 outcome
        groups = {
            "all-correct":    [sid for sid, r in step0_rows.items() if r["correct"] == r["total"]],
            "mixed":          [sid for sid, r in step0_rows.items()
                               if 0 < r["correct"] < r["total"]],
            "all-IDK/inc":    [sid for sid, r in step0_rows.items() if r["correct"] == 0],
        }

        for row_idx, gname in enumerate(group_names):
            gids = groups[gname]
            ax = axes[row_idx][col]
            mean_c, mean_i, mean_n = [], [], []
            for step in STEPS:
                vals_c = [trajs[sid][step]["correct"] / trajs[sid][step]["total"]
                          for sid in gids if sid in trajs and step in trajs[sid]]
                vals_i = [trajs[sid][step]["incorrect"] / trajs[sid][step]["total"]
                          for sid in gids if sid in trajs and step in trajs[sid]]
                vals_n = [trajs[sid][step]["not_attempted"] / trajs[sid][step]["total"]
                          for sid in gids if sid in trajs and step in trajs[sid]]
                mean_c.append(np.mean(vals_c) if vals_c else np.nan)
                mean_i.append(np.mean(vals_i) if vals_i else np.nan)
                mean_n.append(np.mean(vals_n) if vals_n else np.nan)

            xs = list(range(len(STEPS)))
            ax.bar(xs, mean_c, label="correct",       color="#4CAF50", alpha=0.85)
            ax.bar(xs, mean_n, label="not_attempted",  color="#FF9800", alpha=0.85, bottom=mean_c)
            ax.bar(xs, mean_i, label="incorrect",      color="#F44336", alpha=0.85,
                   bottom=[c+n for c,n in zip(mean_c, mean_n)])
            ax.set_xticks(xs); ax.set_xticklabels([f"s{s}" for s in STEPS], fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("avg fraction of 32 samples")
            ax.set_title(f"{exp} | {gname}\n(n={len(gids)} questions)", fontsize=9)
            if row_idx == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    path = f"{OUT_DIR}/fig3_fate_stack.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Pass@k comparison: VAL vs TRAIN, all experiments
# ══════════════════════════════════════════════════════════════════════════════

def fig4_passatk_comparison():
    print("Figure 4: pass@k comparison …")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("pass@k over training steps\nVal set (dashed) vs Train set (solid)", fontsize=13)

    styles = {
        "binary":           ("blue",  "o"),
        "ternary_adaptive": ("red",   "s"),
        "ternary_static":   ("green", "^"),
    }

    for exp, (color, marker) in styles.items():
        exp_base = EXPS[exp]
        for split, ls, label_suffix in [
            ("val_answer",   "--", "VAL"),
            ("train_answer", "-",  "TRAIN"),
        ]:
            xs, ys = [], []
            for step in STEPS:
                d = load_passatk(exp_base, step, split)
                if d is None:
                    continue
                xs.append(step)
                ys.append(d["pass_at_k"]["pass_at_k"])
            if xs:
                ax.plot(xs, ys, color=color, linestyle=ls, marker=marker,
                        label=f"{exp} – {label_suffix}", linewidth=2, markersize=6)

    ax.set_xlabel("global step")
    ax.set_ylabel("pass@k")
    ax.set_ylim(0.3, 0.7)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    path = f"{OUT_DIR}/fig4_passatk_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Deep dive: for questions all-correct at step 0 on VAL,
#             show the DISTRIBUTION of outcomes at each step as violin/box
# ══════════════════════════════════════════════════════════════════════════════

def fig5_allcorrect_violin():
    print("Figure 5: violin plots for all-correct-at-0 questions …")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Distribution of 'fraction correct' across steps\n"
        "For questions that were 100% correct at step 0 (VAL set only)",
        fontsize=12,
    )

    for col, exp in enumerate(["binary", "ternary_adaptive"]):
        ax = axes[col]
        exp_base = EXPS[exp]
        trajs, meta = build_per_sample_trajectory(exp_base, "val_answer")
        if trajs is None:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center", transform=ax.transAxes)
            continue

        all_correct_ids = meta["all_correct_at_0"]
        data_by_step = []
        labels = []
        for step in STEPS:
            vals = [trajs[sid][step]["correct"] / trajs[sid][step]["total"]
                    for sid in all_correct_ids if sid in trajs and step in trajs[sid]]
            data_by_step.append(vals)
            labels.append(f"step {step}\n(n={len(vals)})")

        parts = ax.violinplot(data_by_step, positions=range(len(STEPS)), showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#2196F3")
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("red")
        ax.set_xticks(range(len(STEPS)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("fraction of 32 samples that are 'correct'")
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(f"{exp}\nall-correct at step 0 (n={len(all_correct_ids)} questions)", fontsize=11)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(1, color="gray", linestyle=":", linewidth=0.8)

    plt.tight_layout()
    path = f"{OUT_DIR}/fig5_allcorrect_violin.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Flow: What happens to each "correct" sample?
#             For each originally-correct question, classify its step-480 fate:
#             "still correct", "went to IDK", "went to incorrect", "mixed"
# ══════════════════════════════════════════════════════════════════════════════

def fig6_fate_sankey():
    print("Figure 6: fate classification bar chart (step 0 → step 480) …")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        "Where do step-0 correct questions end up at step 480?\n"
        "(each question categorized by majority outcome at step 480)",
        fontsize=12,
    )

    def classify_fate(traj, sid):
        if 480 not in traj[sid]:
            return "missing"
        s480 = traj[sid][480]
        total = s480["total"]
        c, n, i = s480["correct"], s480["not_attempted"], s480["incorrect"]
        dominant = max(("correct", c), ("IDK", n), ("incorrect", i), key=lambda x: x[1])
        return dominant[0]

    for col, exp in enumerate(["binary", "ternary_adaptive"]):
        ax = axes[col]
        exp_base = EXPS[exp]

        fate_counts = {"VAL": {}, "TRAIN": {}}
        for split_label, split in [("VAL", "val_answer"), ("TRAIN", "train_answer")]:
            trajs, meta = build_per_sample_trajectory(exp_base, split)
            if trajs is None:
                continue
            step0_rows = meta["step0_rows"]
            # Use questions that had ≥1 correct at step 0
            correct_sids = meta["correct_at_0"]
            fates = [classify_fate(trajs, sid) for sid in correct_sids if sid in trajs]
            from collections import Counter
            fate_counts[split_label] = Counter(fates)

        categories = ["correct", "IDK", "incorrect", "mixed", "missing"]
        colors_map  = {"correct": "#4CAF50", "IDK": "#FF9800", "incorrect": "#F44336",
                       "mixed": "#9C27B0", "missing": "#9E9E9E"}
        x = np.arange(2)
        width = 0.15
        offset = -1.5 * width
        for i, cat in enumerate(categories):
            vals = [fate_counts["VAL"].get(cat, 0), fate_counts["TRAIN"].get(cat, 0)]
            ax.bar(x + offset + i * width, vals, width, label=cat, color=colors_map[cat], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(["VAL (held-out)", "TRAIN (GRPO)"])
        ax.set_ylabel("# questions")
        ax.set_title(f"{exp}: fate of step-0 correct questions by step 480", fontsize=10)
        if col == 0:
            ax.legend(fontsize=9)

    plt.tight_layout()
    path = f"{OUT_DIR}/fig6_fate_classification.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Gradual vs sudden degradation:
#            For questions that end up fully IDK at step 480,
#            at which step does the "correct" fraction first drop below 0.5?
# ══════════════════════════════════════════════════════════════════════════════

def fig7_when_degradation_happens():
    print("Figure 7: when does degradation happen? …")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "At which step do originally-correct VAL questions first fall below 50% correct?\n"
        "(only questions that end up ≤25% correct at step 480)",
        fontsize=11,
    )

    for col, exp in enumerate(["binary", "ternary_adaptive"]):
        ax = axes[col]
        exp_base = EXPS[exp]
        trajs, meta = build_per_sample_trajectory(exp_base, "val_answer")
        if trajs is None:
            continue

        first_drop_step = []
        for sid, traj in trajs.items():
            if 0 not in traj or 480 not in traj:
                continue
            if traj[0]["correct"] == 0:  # not correct at 0
                continue
            # Check if degraded by step 480
            frac_480 = traj[480]["correct"] / traj[480]["total"]
            if frac_480 > 0.25:
                continue
            # Find first step where fraction correct < 0.5
            for step in STEPS:
                if step not in traj:
                    continue
                frac = traj[step]["correct"] / traj[step]["total"]
                if frac < 0.5:
                    first_drop_step.append(step)
                    break

        from collections import Counter
        counts = Counter(first_drop_step)
        xs = sorted(counts.keys())
        ys = [counts[x] for x in xs]
        ax.bar(xs, ys, width=60, color="#E53935", alpha=0.8)
        ax.set_xticks(xs)
        ax.set_xlabel("first step where correct fraction < 50%")
        ax.set_ylabel("# questions")
        ax.set_title(f"{exp} (n={len(first_drop_step)} degraded questions)", fontsize=10)

    plt.tight_layout()
    path = f"{OUT_DIR}/fig7_when_degradation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 – Example questions: show full trajectory for 6 degraded and 6 stable
# ══════════════════════════════════════════════════════════════════════════════

def fig8_example_trajectories():
    print("Figure 8: example trajectories for selected questions …")

    exp = "ternary_adaptive"
    exp_base = EXPS[exp]
    trajs, meta = build_per_sample_trajectory(exp_base, "val_answer")
    step0_rows = meta["step0_rows"]

    # Degraded: all-correct at 0, fully IDK (=0 correct) at 480
    degraded = [sid for sid in meta["all_correct_at_0"]
                if 480 in trajs[sid] and trajs[sid][480]["correct"] == 0]
    # Stable: all-correct at 0, still mostly correct at 480
    stable = [sid for sid in meta["all_correct_at_0"]
              if 480 in trajs[sid] and trajs[sid][480]["correct"] >= 24]

    n_show = min(6, min(len(degraded), len(stable)))
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    fig.suptitle(
        f"ternary_adaptive – example question trajectories (VAL set)\n"
        f"Top: degraded (was 100% correct → went IDK)  |  Bottom: stable (stayed correct)",
        fontsize=11,
    )

    def plot_traj(ax, sid, title_color):
        traj = trajs[sid]
        xs = sorted(traj.keys())
        ys_c = [traj[s]["correct"] / traj[s]["total"] for s in xs]
        ys_n = [traj[s]["not_attempted"] / traj[s]["total"] for s in xs]
        ys_i = [traj[s]["incorrect"] / traj[s]["total"] for s in xs]

        ax.stackplot(xs, ys_c, ys_n, ys_i,
                     labels=["correct", "IDK", "incorrect"],
                     colors=["#4CAF50", "#FF9800", "#F44336"], alpha=0.8)
        q = step0_rows[sid]["question"]
        ax.set_title(f"Q: {q[:45]}…" if len(q) > 45 else q, fontsize=7, color=title_color)
        ax.set_xlabel("step", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(xs)
        ax.tick_params(labelsize=6)

    for j, sid in enumerate(degraded[:n_show]):
        plot_traj(axes[0][j], sid, "darkred")
        if j == 0:
            axes[0][j].set_ylabel("fraction of 32 responses", fontsize=8)

    for j, sid in enumerate(stable[:n_show]):
        plot_traj(axes[1][j], sid, "darkgreen")
        if j == 0:
            axes[1][j].set_ylabel("fraction of 32 responses", fontsize=8)
            axes[1][j].legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    path = f"{OUT_DIR}/fig8_example_trajectories.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CSV EXPORT – per-sample fate summary table
# ══════════════════════════════════════════════════════════════════════════════

def export_csv():
    print("Exporting CSV …")
    rows = []
    for exp in ["binary", "ternary_adaptive"]:
        for split_label, split in [("val", "val_answer"), ("train", "train_answer")]:
            exp_base = EXPS[exp]
            trajs, meta = build_per_sample_trajectory(exp_base, split)
            if trajs is None:
                continue
            for sid, traj in trajs.items():
                row = {"exp": exp, "split": split_label, "id": sid,
                       "question": meta["step0_rows"][sid]["question"][:80]}
                for step in STEPS:
                    if step in traj:
                        t = traj[step]
                        row[f"s{step}_correct"] = t["correct"]
                        row[f"s{step}_idk"]     = t["not_attempted"]
                        row[f"s{step}_inc"]      = t["incorrect"]
                        row[f"s{step}_frac_correct"] = t["correct"] / t["total"]
                rows.append(row)

    df = pd.DataFrame(rows)
    path = f"{OUT_DIR}/per_sample_trajectories.csv"
    df.to_csv(path, index=False)
    print(f"  saved → {path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Refusal Degradation Analysis")
    print("=" * 70)
    fig1_aggregate_metrics()
    fig2_persample_fate()
    fig3_fate_stack()
    fig4_passatk_comparison()
    fig5_allcorrect_violin()
    fig6_fate_sankey()
    fig7_when_degradation_happens()
    fig8_example_trajectories()
    df = export_csv()
    print()
    print("All figures saved to:", OUT_DIR)
    print("CSV saved with", len(df), "rows")
