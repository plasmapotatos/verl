"""
Figures 14-16: Incorrect → IDK transition analysis.

Companion to analyze_not_attempted_confidence.py (correct→IDK).

Key question: When a question that was INCORRECT at baseline becomes IDK by step 480,
is this a "good" (model learning genuine uncertainty) or "bad" (over-refusing) signal?
And do the patterns differ by reward mode?

Approach:
  For each experiment / val set, we look at ALL questions (not just correct-at-baseline ones)
  and classify each question's trajectory type:

    "correct→IDK"   : had ≥1 correct at baseline, 0 correct at step 480, IDK increased
    "incorrect→IDK" : had 0 correct at baseline, >0 incorrect at baseline,
                      IDK fraction at step 480 > IDK fraction at baseline
    "incorrect→stay": had 0 correct at baseline, >0 incorrect, IDK did NOT increase
    "idk→idk"       : already mostly IDK at baseline

Figures:
  Fig 14: Stacked bar — how many questions fall into each trajectory type per reward mode?
          Compare binary vs ternary_adaptive vs ternary_static.
  Fig 15: For "incorrect→IDK" questions: histogram of baseline frac_correct (always 0 for
          this group) and baseline frac_incorrect — i.e. how wrong were they? Are they
          high-confidence-wrong (nearly all incorrect, model was sure but wrong) or
          low-confidence-wrong (split between IDK/incorrect)?
  Fig 16: Mean IDK trajectory over steps for "incorrect→IDK" questions, split by
          baseline frac_incorrect (>75%, 50-75%, 25-50%, <25%), for binary vs ternary_adaptive.
          Shows whether ternary converts "confident wrongs" differently than "uncertain wrongs".
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths (shared with analyze_not_attempted_confidence.py) ──────────────────
BASE_ROOT = "/work/hdd/bbsg/twei2/rl/verl/outputs/rl/simpleqa_rich_sft_grpo_refusal"
EXPS = {
    "binary":           f"{BASE_ROOT}/simpleqa_rich_sft_grpo_refusal_binary",
    "ternary_adaptive": f"{BASE_ROOT}/simpleqa_rich_sft_grpo_refusal_ternary_adaptive",
    "ternary_static":   f"{BASE_ROOT}/simpleqa_rich_sft_grpo_refusal_ternary_static",
}
STEPS = [0, 100, 200, 300, 400, 480]
SETS = {
    "val_answer":   "train_richqa_answer_origqa_frac0.2",
    "train_answer": "train_richqa_answer_origqa_frac0.8_frac0.2",
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

EXP_COLORS = {
    "binary":           "#1565C0",
    "ternary_adaptive": "#C62828",
    "ternary_static":   "#2E7D32",
}

# ── Core helpers (copied from analyze_not_attempted_confidence.py) ────────────

def load_eval(exp_base, step, set_key):
    path = f"{exp_base}/global_step_{step}/pass@k/{SETS[set_key]}/eval.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def row_summary(row):
    evals = [g["evaluation"] for g in row["graders"]]
    return {
        "id":            row["id"],
        "question":      row["question"],
        "correct":       evals.count("correct"),
        "incorrect":     evals.count("incorrect"),
        "not_attempted": evals.count("not_attempted"),
        "total":         len(evals),
    }


def build_all_sample_trajectories(exp_base, split, steps=None):
    """
    Like build_per_sample_trajectory but returns ALL questions (not just correct-at-baseline).

    Returns
    -------
    trajectories : dict  {sample_id: {step: row_summary_dict}}
    meta         : dict  with 'baseline_step', 'baseline_rows' (all sids → row_summary at baseline)
    """
    if steps is None:
        steps = STEPS

    baseline_step = None
    baseline_data = None
    for s in steps:
        d = load_eval(exp_base, s, split)
        if d is not None:
            baseline_step = s
            baseline_data = d
            break

    if baseline_data is None:
        return None, None

    baseline_rows = {r["id"]: row_summary(r) for r in baseline_data["rows"]}
    all_sids = set(baseline_rows.keys())

    trajectories = defaultdict(dict)
    for step in steps:
        d = load_eval(exp_base, step, split)
        if d is None:
            continue
        for row in d["rows"]:
            sid = row["id"]
            if sid in all_sids:
                trajectories[sid][step] = row_summary(row)

    meta = {
        "baseline_step": baseline_step,
        "baseline_rows": baseline_rows,
    }
    return dict(trajectories), meta


def classify_trajectory(traj, baseline_rows, final_step=480, baseline_step=0):
    """
    Classify a single question's trajectory into one of five categories.

    Parameters
    ----------
    traj : dict  {step: row_summary}
    baseline_rows : dict  {sid: row_summary at baseline}

    Returns one of:
      "correct→IDK"    had correct at baseline, 0 correct at final, IDK increased
      "correct→stay"   had correct at baseline, still correct at final
      "incorrect→IDK"  0 correct at baseline, >0 incorrect, IDK fraction grew significantly
      "incorrect→stay" 0 correct at baseline, >0 incorrect, IDK did NOT grow
      "idk_dominated"  already ≥75% IDK at baseline
    """
    if baseline_step not in traj or final_step not in traj:
        return "incomplete"

    b = traj[baseline_step]
    f = traj[final_step]
    total = b["total"]

    frac_correct_b  = b["correct"]  / total
    frac_idk_b      = b["not_attempted"] / total
    frac_incorrect_b = b["incorrect"] / total

    frac_idk_f      = f["not_attempted"] / total
    frac_correct_f  = f["correct"]  / total

    if frac_idk_b >= 0.75:
        return "idk_dominated"

    if frac_correct_b > 0:
        if frac_correct_f == 0:
            return "correct→IDK"
        else:
            return "correct→stay"

    # No correct at baseline
    if frac_incorrect_b == 0:
        return "idk_dominated"  # no correct, no incorrect → already IDK

    # Had incorrect answers, no correct
    idk_increase = frac_idk_f - frac_idk_b
    if idk_increase > 0.2:   # IDK grew by at least 20 percentage points
        return "incorrect→IDK"
    else:
        return "incorrect→stay"


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY: print trajectory classification counts for all experiments
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    print("=" * 70)
    print("TRAJECTORY CLASSIFICATION SUMMARY (val_answer set)")
    print("=" * 70)

    categories = ["correct→IDK", "correct→stay", "incorrect→IDK",
                  "incorrect→stay", "idk_dominated", "incomplete"]

    for exp in ["binary", "ternary_adaptive", "ternary_static"]:
        trajs, meta = build_all_sample_trajectories(EXPS[exp], "val_answer")
        if trajs is None:
            print(f"{exp}: no data\n")
            continue

        baseline_step = meta["baseline_step"]
        final_step = max(s for t in trajs.values() for s in t.keys())

        counts = defaultdict(int)
        for sid, traj in trajs.items():
            cat = classify_trajectory(traj, meta["baseline_rows"],
                                      final_step=final_step,
                                      baseline_step=baseline_step)
            counts[cat] += 1

        total = sum(counts.values())
        print(f"\n{exp}  (baseline=step {baseline_step}, final=step {final_step}):")
        for cat in categories:
            n = counts[cat]
            print(f"  {cat:<20s}: {n:3d} ({100*n/total:.1f}%)")

        # For incorrect→IDK: what was their baseline frac_incorrect?
        inc_to_idk = [
            sid for sid, traj in trajs.items()
            if classify_trajectory(traj, meta["baseline_rows"],
                                   final_step=final_step,
                                   baseline_step=baseline_step) == "incorrect→IDK"
        ]
        if inc_to_idk:
            fracs = [meta["baseline_rows"][sid]["incorrect"] /
                     meta["baseline_rows"][sid]["total"]
                     for sid in inc_to_idk]
            idk_fracs = [meta["baseline_rows"][sid]["not_attempted"] /
                         meta["baseline_rows"][sid]["total"]
                         for sid in inc_to_idk]
            print(f"\n  incorrect→IDK detail (n={len(inc_to_idk)}):")
            print(f"    baseline frac_incorrect: mean={np.mean(fracs):.3f}, "
                  f"median={np.median(fracs):.3f}")
            print(f"    baseline frac_idk:       mean={np.mean(idk_fracs):.3f}, "
                  f"median={np.median(idk_fracs):.3f}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 14 – Stacked bar: trajectory type counts per reward mode
# ══════════════════════════════════════════════════════════════════════════════

def fig14_trajectory_type_counts():
    print("Figure 14: trajectory type counts per reward mode …")

    exps = ["binary", "ternary_adaptive", "ternary_static"]
    categories = ["correct→IDK", "incorrect→IDK", "correct→stay",
                  "incorrect→stay", "idk_dominated"]
    cat_colors  = {
        "correct→IDK":    "#B71C1C",   # dark red
        "incorrect→IDK":  "#FF7043",   # orange-red
        "correct→stay":   "#388E3C",   # green
        "incorrect→stay": "#81C784",   # light green
        "idk_dominated":  "#9E9E9E",   # grey
    }

    all_counts = {}
    for exp in exps:
        trajs, meta = build_all_sample_trajectories(EXPS[exp], "val_answer")
        if trajs is None:
            all_counts[exp] = {}
            continue
        baseline_step = meta["baseline_step"]
        final_step = max(s for t in trajs.values() for s in t.keys())
        counts = defaultdict(int)
        for sid, traj in trajs.items():
            cat = classify_trajectory(traj, meta["baseline_rows"],
                                      final_step=final_step,
                                      baseline_step=baseline_step)
            counts[cat] += 1
        all_counts[exp] = dict(counts)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Question trajectory types across all val questions\n"
        "IDK-growth threshold: IDK fraction increases by ≥20 pp from baseline to step 480",
        fontsize=12, fontweight="bold",
    )

    x = np.arange(len(exps))
    width = 0.5
    bottoms = np.zeros(len(exps))

    for cat in categories:
        vals = [all_counts[exp].get(cat, 0) for exp in exps]
        bars = ax.bar(x, vals, width, bottom=bottoms,
                      label=cat, color=cat_colors[cat], alpha=0.88)
        # Label bars that are tall enough
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 5:
                ax.text(xi, b + v / 2, str(v), ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(exps, fontsize=11)
    ax.set_ylabel("# questions")
    ax.set_xlabel("reward mode")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(bottoms) * 1.12)

    path = f"{OUT_DIR}/fig14_trajectory_type_counts.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 15 – Baseline confidence of "incorrect→IDK" questions
#             Are these confident-wrong or uncertain-wrong at baseline?
# ══════════════════════════════════════════════════════════════════════════════

def fig15_incorrect_to_idk_baseline():
    print("Figure 15: baseline breakdown of incorrect→IDK questions …")

    exps = ["binary", "ternary_adaptive", "ternary_static"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Baseline composition of questions that went from INCORRECT → IDK\n"
        "(each bar = one question, sorted by baseline frac_incorrect)",
        fontsize=12, fontweight="bold",
    )

    for col, exp in enumerate(exps):
        ax = axes[col]
        trajs, meta = build_all_sample_trajectories(EXPS[exp], "val_answer")

        if trajs is None:
            ax.text(0.5, 0.5, f"{exp}\nno val data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_title(exp, fontsize=11)
            continue

        baseline_step = meta["baseline_step"]
        final_step = max(s for t in trajs.values() for s in t.keys())

        inc_to_idk_sids = [
            sid for sid, traj in trajs.items()
            if classify_trajectory(traj, meta["baseline_rows"],
                                   final_step=final_step,
                                   baseline_step=baseline_step) == "incorrect→IDK"
        ]

        if not inc_to_idk_sids:
            ax.text(0.5, 0.5, f"{exp}\nno incorrect→IDK questions",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray")
            ax.set_title(f"{exp} (baseline=step {baseline_step})", fontsize=10)
            continue

        # Sort by baseline frac_incorrect (descending = most confidently wrong first)
        def baseline_frac_inc(sid):
            b = meta["baseline_rows"][sid]
            return b["incorrect"] / b["total"]

        inc_to_idk_sids.sort(key=baseline_frac_inc, reverse=True)

        fracs_inc = [meta["baseline_rows"][sid]["incorrect"] /
                     meta["baseline_rows"][sid]["total"]
                     for sid in inc_to_idk_sids]
        fracs_idk = [meta["baseline_rows"][sid]["not_attempted"] /
                     meta["baseline_rows"][sid]["total"]
                     for sid in inc_to_idk_sids]
        fracs_cor = [meta["baseline_rows"][sid]["correct"] /
                     meta["baseline_rows"][sid]["total"]
                     for sid in inc_to_idk_sids]

        xs = np.arange(len(inc_to_idk_sids))
        ax.bar(xs, fracs_inc, color="#F44336", alpha=0.85, label="incorrect")
        ax.bar(xs, fracs_idk, color="#FF9800", alpha=0.85,
               bottom=fracs_inc, label="IDK")
        ax.bar(xs, fracs_cor, color="#4CAF50", alpha=0.85,
               bottom=[i + d for i, d in zip(fracs_inc, fracs_idk)], label="correct")

        mean_inc = np.mean(fracs_inc)
        mean_idk = np.mean(fracs_idk)
        ax.axhline(mean_inc, color="#B71C1C", linestyle="--", linewidth=1.5,
                   label=f"mean frac_incorrect={mean_inc:.2f}")
        ax.axhline(mean_inc + mean_idk, color="#E65100", linestyle=":",
                   linewidth=1.2, label=f"mean frac_idk={mean_idk:.2f}")

        ax.set_title(
            f"{exp}  (baseline=step {baseline_step})\n"
            f"n={len(inc_to_idk_sids)} incorrect→IDK questions",
            fontsize=10,
        )
        ax.set_xlabel("question (sorted by baseline frac_incorrect ↓)", fontsize=9)
        ax.set_ylabel("fraction of 32 samples at baseline", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    path = f"{OUT_DIR}/fig15_incorrect_to_idk_baseline.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 16 – IDK trajectory over steps for incorrect→IDK questions,
#             binned by baseline frac_incorrect
# ══════════════════════════════════════════════════════════════════════════════

def fig16_incorrect_to_idk_trajectory():
    print("Figure 16: IDK trajectory for incorrect→IDK questions by confidence bin …")

    exps = ["binary", "ternary_adaptive"]   # ternary_static only from step 100
    bins = [
        ("confident-wrong\n(>75% incorrect)",  lambda f: f > 0.75),
        ("mostly-wrong\n(50–75%)",             lambda f: 0.50 < f <= 0.75),
        ("mixed\n(25–50%)",                    lambda f: 0.25 < f <= 0.50),
        ("mostly-IDK\n(<25% incorrect)",       lambda f: f <= 0.25),
    ]
    bin_colors = ["#B71C1C", "#EF5350", "#FF8A65", "#FFCCBC"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(
        "IDK fraction trajectory for questions that went from INCORRECT → IDK\n"
        "Binned by baseline fraction-incorrect (how confident-wrong they were)",
        fontsize=12, fontweight="bold",
    )

    for col, exp in enumerate(exps):
        ax = axes[col]
        trajs, meta = build_all_sample_trajectories(EXPS[exp], "val_answer")

        if trajs is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(exp)
            continue

        baseline_step = meta["baseline_step"]
        final_step = max(s for t in trajs.values() for s in t.keys())
        available_steps = sorted({s for t in trajs.values() for s in t.keys()})

        inc_to_idk_sids = [
            sid for sid, traj in trajs.items()
            if classify_trajectory(traj, meta["baseline_rows"],
                                   final_step=final_step,
                                   baseline_step=baseline_step) == "incorrect→IDK"
        ]

        def baseline_frac_inc(sid):
            b = meta["baseline_rows"][sid]
            return b["incorrect"] / b["total"]

        for (bin_label, bin_fn), color in zip(bins, bin_colors):
            group = [sid for sid in inc_to_idk_sids if bin_fn(baseline_frac_inc(sid))]
            if not group:
                continue

            mean_idk, std_idk = [], []
            xs_used = []
            for step in available_steps:
                vals = [trajs[sid][step]["not_attempted"] / trajs[sid][step]["total"]
                        for sid in group if step in trajs[sid]]
                if not vals:
                    continue
                mean_idk.append(np.mean(vals))
                std_idk.append(np.std(vals))
                xs_used.append(step)

            ax.plot(xs_used, mean_idk, color=color, linewidth=2.5,
                    marker="o", markersize=5, label=f"{bin_label} (n={len(group)})")
            lo = [m - s for m, s in zip(mean_idk, std_idk)]
            hi = [m + s for m, s in zip(mean_idk, std_idk)]
            ax.fill_between(xs_used, lo, hi, color=color, alpha=0.15)

        ax.set_title(f"{exp}\n(baseline=step {baseline_step})", fontsize=11)
        ax.set_xlabel("training step")
        ax.set_ylabel("mean IDK fraction (±1 std)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(available_steps)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25)

    path = f"{OUT_DIR}/fig16_incorrect_to_idk_trajectory.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Incorrect → IDK Transition Analysis (Figs 14–16)")
    print("=" * 70)
    print_summary()
    fig14_trajectory_type_counts()
    fig15_incorrect_to_idk_baseline()
    fig16_incorrect_to_idk_trajectory()
    print()
    print("All figures saved to:", OUT_DIR)
