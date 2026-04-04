"""
Figures 10-13: Confidence-based analysis of refusal degradation.

Key story:
  - Binary: only ~5% of questions degrade (correct at step 0 → IDK at step 480).
    Those that DO degrade were already low-confidence at baseline (mean frac_correct=0.14).
  - Ternary_adaptive: ~36% of questions degrade.
    Those that degrade were HIGH-confidence at baseline (mean frac_correct=0.62) —
    they were genuinely known answers that the model was "trained" to refuse.

Fig 10: Confidence distribution (step-0 frac_correct) of degraded vs non-degraded.
Fig 11: Baseline breakdown for ternary_adaptive degraded questions
        (stacked bar of correct/IDK/incorrect by confidence bin at step 0).
Fig 12: Mean trajectory (correct fraction over training steps) binned by step-0 confidence.
Fig 13: 2x2 — Binary vs Ternary_adaptive × low vs high confidence;
        shows that binary PRESERVES high-confidence answers, ternary DEGRADES them.
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

# ── Paths ──────────────────────────────────────────────────────────────────────
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
os.makedirs(OUT_DIR, exist_ok=True)

# Colour palette
EXP_COLORS = {
    "binary":           "#1565C0",   # blue
    "ternary_adaptive": "#C62828",   # red
    "ternary_static":   "#2E7D32",   # green
}
DEGRADED_COLOR     = "#B71C1C"   # dark red / saturated
NONDEGRADED_COLOR  = "#90CAF9"   # light blue / pastel

# ── Core helpers ───────────────────────────────────────────────────────────────

def load_eval(exp_base, step, set_key):
    path = f"{exp_base}/global_step_{step}/pass@k/{SETS[set_key]}/eval.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def row_summary(row):
    """Return per-question counts of correct/incorrect/not_attempted across k=32 samples."""
    evals = [g["evaluation"] for g in row["graders"]]
    return {
        "id":            row["id"],
        "question":      row["question"],
        "correct":       evals.count("correct"),
        "incorrect":     evals.count("incorrect"),
        "not_attempted": evals.count("not_attempted"),
        "total":         len(evals),
    }


def build_per_sample_trajectory(exp_base, split, steps=None):
    """
    Build per-sample trajectory over all available steps.

    Returns
    -------
    trajectories : dict  {sample_id: {step: row_summary_dict}}
                   Only includes samples with >=1 correct response at the *first available* step.
    meta         : dict  with 'baseline_step', 'step0_rows' (keyed by sample_id),
                         'correct_at_baseline', 'all_correct_at_baseline'
    """
    if steps is None:
        steps = STEPS

    # Find the first available step
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
    correct_at_baseline     = {sid for sid, s in baseline_rows.items() if s["correct"] > 0}
    all_correct_at_baseline = {sid for sid, s in baseline_rows.items()
                               if s["correct"] == s["total"]}

    trajectories = defaultdict(dict)
    for step in steps:
        d = load_eval(exp_base, step, split)
        if d is None:
            continue
        for row in d["rows"]:
            sid = row["id"]
            if sid in correct_at_baseline:
                trajectories[sid][step] = row_summary(row)

    meta = {
        "baseline_step":          baseline_step,
        "step0_rows":             baseline_rows,   # includes ALL rows (not just correct-at-baseline)
        "correct_at_baseline":    correct_at_baseline,
        "all_correct_at_baseline": all_correct_at_baseline,
    }
    return dict(trajectories), meta


def classify_degraded(trajectories, meta):
    """
    For each sample in trajectories:
      degraded     = had >=1 correct at baseline, ends at 0 correct at final available step
      non_degraded = had >=1 correct at baseline, ends at >=1 correct at final step

    Returns two sets of sample IDs.
    """
    final_step = max(STEPS)
    degraded     = set()
    non_degraded = set()
    for sid, traj in trajectories.items():
        steps_avail = sorted(traj.keys())
        if not steps_avail:
            continue
        last = steps_avail[-1]
        if last != final_step:
            continue   # incomplete trajectory; skip
        if traj[last]["correct"] == 0:
            degraded.add(sid)
        else:
            non_degraded.add(sid)
    return degraded, non_degraded


def get_baseline_frac_correct(sid, trajectories, meta):
    """Fraction of 32 samples that were correct at the baseline step."""
    step = meta["baseline_step"]
    t = trajectories.get(sid, {}).get(step)
    if t is None:
        return np.nan
    return t["correct"] / t["total"]


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 – Confidence distribution of degraded vs non-degraded questions
# ══════════════════════════════════════════════════════════════════════════════

def fig10_confidence_distribution():
    print("Figure 10: confidence distribution of degraded vs non-degraded …")

    exps_to_plot = ["binary", "ternary_adaptive", "ternary_static"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Distribution of baseline pass@k confidence for degraded vs non-degraded questions\n"
        "(degraded = ≥1 correct at baseline → 0 correct at step 480)",
        fontsize=13, fontweight="bold",
    )

    bins = np.linspace(0, 1, 17)   # 16 bins from 0 to 1

    for col, exp in enumerate(exps_to_plot):
        ax = axes[col]
        exp_base = EXPS[exp]
        trajs, meta = build_per_sample_trajectory(exp_base, "val_answer")

        if trajs is None:
            ax.text(0.5, 0.5, f"{exp}\nval data unavailable",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_title(exp, fontsize=11)
            continue

        degraded, non_degraded = classify_degraded(trajs, meta)

        # Baseline fractions
        frac_degraded     = [get_baseline_frac_correct(sid, trajs, meta) for sid in degraded]
        frac_non_degraded = [get_baseline_frac_correct(sid, trajs, meta) for sid in non_degraded]
        frac_degraded     = [v for v in frac_degraded     if not np.isnan(v)]
        frac_non_degraded = [v for v in frac_non_degraded if not np.isnan(v)]

        baseline_label = "step 0" if meta["baseline_step"] == 0 else f"step {meta['baseline_step']}"

        ax.hist(frac_non_degraded, bins=bins, color=NONDEGRADED_COLOR, alpha=0.8,
                label=f"Non-degraded (n={len(frac_non_degraded)})", edgecolor="white")
        ax.hist(frac_degraded, bins=bins, color=DEGRADED_COLOR, alpha=0.8,
                label=f"Degraded (n={len(frac_degraded)})", edgecolor="white")

        # Annotate means
        if frac_degraded:
            md = np.mean(frac_degraded)
            ax.axvline(md, color=DEGRADED_COLOR, linestyle="--", linewidth=1.8,
                       label=f"Degraded mean={md:.2f}")
        if frac_non_degraded:
            mn = np.mean(frac_non_degraded)
            ax.axvline(mn, color=NONDEGRADED_COLOR, linestyle="--", linewidth=1.8,
                       label=f"Non-deg mean={mn:.2f}")

        n_total = len(frac_degraded) + len(frac_non_degraded)
        pct_deg = 100 * len(frac_degraded) / n_total if n_total else 0
        title_note = f"(baseline = {baseline_label})" if meta["baseline_step"] != 0 else ""

        ax.set_title(
            f"{exp} {title_note}\n"
            f"{len(frac_degraded)}/{n_total} questions degrade ({pct_deg:.1f}%)",
            fontsize=10,
        )
        ax.set_xlabel(f"Fraction of 32 samples correct at {baseline_label}")
        ax.set_ylabel("Number of questions")
        ax.legend(fontsize=8)
        ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    path = f"{OUT_DIR}/fig10_confidence_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11 – Baseline breakdown for ternary_adaptive degraded questions
#             Stacked bar: correct / idk / incorrect at step 0 per confidence bin
# ══════════════════════════════════════════════════════════════════════════════

def fig11_baseline_breakdown():
    print("Figure 11: baseline breakdown of ternary_adaptive degraded questions …")

    exp = "ternary_adaptive"
    exp_base = EXPS[exp]
    trajs, meta = build_per_sample_trajectory(exp_base, "val_answer")

    degraded, _ = classify_degraded(trajs, meta)

    # Confidence bins based on fraction-correct at baseline
    bin_edges  = [0.0, 0.25, 0.50, 0.75, 1.01]
    bin_labels = ["0–25%", "25–50%", "50–75%", "75–100%"]

    # Group degraded questions by confidence bin
    bins_data = {label: [] for label in bin_labels}
    for sid in degraded:
        frac = get_baseline_frac_correct(sid, trajs, meta)
        if np.isnan(frac):
            continue
        for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if lo <= frac < hi:
                bins_data[bin_labels[i]].append(sid)
                break

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(
        "ternary_adaptive — Baseline composition of questions that later degrade\n"
        "Each bar = one confidence bin (% correct at step 0). "
        "Stacked: correct / IDK / incorrect at step 0.",
        fontsize=12, fontweight="bold",
    )

    baseline_step = meta["baseline_step"]
    x_pos = np.arange(len(bin_labels))
    bar_width = 0.55

    mean_corrects     = []
    mean_idks         = []
    mean_incorrects   = []
    ns                = []

    for label in bin_labels:
        sids = bins_data[label]
        ns.append(len(sids))
        if not sids:
            mean_corrects.append(0)
            mean_idks.append(0)
            mean_incorrects.append(0)
            continue
        fracs_c = [trajs[sid][baseline_step]["correct"]       / trajs[sid][baseline_step]["total"] for sid in sids]
        fracs_n = [trajs[sid][baseline_step]["not_attempted"] / trajs[sid][baseline_step]["total"] for sid in sids]
        fracs_i = [trajs[sid][baseline_step]["incorrect"]     / trajs[sid][baseline_step]["total"] for sid in sids]
        mean_corrects.append(np.mean(fracs_c))
        mean_idks.append(np.mean(fracs_n))
        mean_incorrects.append(np.mean(fracs_i))

    bars_c = ax.bar(x_pos, mean_corrects, bar_width,
                    label="correct",       color="#4CAF50", alpha=0.9)
    bars_n = ax.bar(x_pos, mean_idks, bar_width,
                    label="not_attempted", color="#FF9800", alpha=0.9,
                    bottom=mean_corrects)
    bottom_ni = [c + n for c, n in zip(mean_corrects, mean_idks)]
    bars_i = ax.bar(x_pos, mean_incorrects, bar_width,
                    label="incorrect",     color="#F44336", alpha=0.9,
                    bottom=bottom_ni)

    # Annotate n= on each bar
    for i, (xi, n) in enumerate(zip(x_pos, ns)):
        ax.text(xi, 1.02, f"n={n}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{label}\n(n={n})" for label, n in zip(bin_labels, ns)],
        fontsize=10,
    )
    ax.set_xlabel("Step-0 confidence bin (fraction of 32 samples that were correct)", fontsize=11)
    ax.set_ylabel("Average fraction of 32 samples at step 0", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10)
    ax.set_title(
        f"Even high-confidence (75–100%) questions had mostly correct answers at step 0\n"
        f"— yet all these groups degrade to IDK by step 480 in ternary_adaptive",
        fontsize=10,
    )

    plt.tight_layout()
    path = f"{OUT_DIR}/fig11_baseline_breakdown.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12 – Trajectory of degraded questions binned by step-0 confidence
#             Subplot 1: ternary_adaptive (main story)
#             Subplot 2: binary (comparison, few degraded cases)
# ══════════════════════════════════════════════════════════════════════════════

def fig12_trajectory_by_confidence():
    print("Figure 12: trajectories binned by step-0 confidence …")

    bin_defs = [
        ("<25%",   0.00, 0.25, "#9C27B0"),   # purple
        ("25–75%", 0.25, 0.75, "#FF9800"),   # orange
        (">75%",   0.75, 1.01, "#E53935"),   # red
    ]

    exps_to_show = [
        ("ternary_adaptive", "ternary_adaptive — ALL questions with ≥1 correct at step 0",
         "Even the >75%-correct-at-step-0 group fully degrades to IDK"),
        ("binary",           "binary — ALL questions with ≥1 correct at step 0",
         "High-confidence questions are PRESERVED; only already-low-confidence ones degrade"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(
        "Correct-fraction trajectory over GRPO training, binned by step-0 confidence\n"
        "(val set; shading = ±1 std across questions)",
        fontsize=13, fontweight="bold",
    )

    for col, (exp, title, subtitle) in enumerate(exps_to_show):
        ax = axes[col]
        exp_base = EXPS[exp]
        trajs, meta = build_per_sample_trajectory(exp_base, "val_answer")
        if trajs is None:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center", transform=ax.transAxes)
            continue

        baseline_step = meta["baseline_step"]
        all_steps = sorted({s for traj in trajs.values() for s in traj.keys()})

        # Bin ALL questions with >=1 correct at baseline by their baseline confidence
        for bin_label, lo, hi, color in bin_defs:
            bin_sids = [
                sid for sid in trajs
                if lo <= get_baseline_frac_correct(sid, trajs, meta) < hi
            ]
            if not bin_sids:
                continue

            mean_traj = []
            std_traj  = []
            steps_plot = []
            for step in all_steps:
                vals = [
                    trajs[sid][step]["correct"] / trajs[sid][step]["total"]
                    for sid in bin_sids if step in trajs[sid]
                ]
                if vals:
                    mean_traj.append(np.mean(vals))
                    std_traj.append(np.std(vals))
                    steps_plot.append(step)

            mean_arr = np.array(mean_traj)
            std_arr  = np.array(std_traj)
            ax.plot(steps_plot, mean_arr, color=color, linewidth=2.5, marker="o",
                    label=f"Step-0 conf {bin_label} (n={len(bin_sids)})")
            ax.fill_between(steps_plot,
                            np.clip(mean_arr - std_arr, 0, 1),
                            np.clip(mean_arr + std_arr, 0, 1),
                            color=color, alpha=0.18)

        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_xlabel("Global training step", fontsize=11)
        ax.set_ylabel("Mean fraction of 32 samples correct", fontsize=11)
        ax.set_xticks(all_steps)
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(1, color="gray", linestyle=":", linewidth=0.8)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = f"{OUT_DIR}/fig12_trajectory_by_confidence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 13 – 2×2 summary: Binary vs Ternary × Low vs High confidence
#             Rows: binary, ternary_adaptive
#             Cols: low confidence (<50%), high confidence (>=50%)
#             Only questions with >=1 correct at step 0
#             Each cell: stacked area / lines of correct/IDK/incorrect over steps
# ══════════════════════════════════════════════════════════════════════════════

def fig13_binary_vs_ternary_protection():
    print("Figure 13: 2x2 binary vs ternary × low/high confidence …")

    exps_rows = [
        ("binary",           "Binary reward", "#1565C0"),
        ("ternary_adaptive", "Ternary adaptive reward", "#C62828"),
    ]
    conf_cols = [
        ("Low confidence (<50%)",  0.00, 0.50),
        ("High confidence (≥50%)", 0.50, 1.01),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Does binary reward protect high-confidence correct answers from becoming IDK?\n"
        "Only questions with ≥1 correct at step 0 (val set)",
        fontsize=13, fontweight="bold",
    )

    # Shared legend handles (built on first cell)
    legend_handles = None

    for row_idx, (exp, exp_label, exp_color) in enumerate(exps_rows):
        exp_base = EXPS[exp]
        trajs, meta = build_per_sample_trajectory(exp_base, "val_answer")
        if trajs is None:
            for col_idx in range(2):
                ax = axes[row_idx][col_idx]
                ax.text(0.5, 0.5, "unavailable", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="gray")
            continue

        baseline_step = meta["baseline_step"]
        all_steps = sorted({s for traj in trajs.values() for s in traj.keys()})

        for col_idx, (conf_label, lo, hi) in enumerate(conf_cols):
            ax = axes[row_idx][col_idx]

            # Filter to questions with >=1 correct at baseline AND in this confidence bin
            bin_sids = [
                sid for sid in trajs
                if lo <= get_baseline_frac_correct(sid, trajs, meta) < hi
            ]

            if not bin_sids:
                ax.text(0.5, 0.5, "no questions in this bin", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
                ax.set_title(f"{exp_label}\n{conf_label}", fontsize=10)
                continue

            mean_c, mean_n, mean_i = [], [], []
            steps_plot = []
            for step in all_steps:
                vals_c = [trajs[sid][step]["correct"]       / trajs[sid][step]["total"]
                          for sid in bin_sids if step in trajs[sid]]
                vals_n = [trajs[sid][step]["not_attempted"] / trajs[sid][step]["total"]
                          for sid in bin_sids if step in trajs[sid]]
                vals_i = [trajs[sid][step]["incorrect"]     / trajs[sid][step]["total"]
                          for sid in bin_sids if step in trajs[sid]]
                if vals_c:
                    mean_c.append(np.mean(vals_c))
                    mean_n.append(np.mean(vals_n))
                    mean_i.append(np.mean(vals_i))
                    steps_plot.append(step)

            mean_c = np.array(mean_c)
            mean_n = np.array(mean_n)
            mean_i = np.array(mean_i)

            # Stacked filled area: correct / IDK / incorrect
            h_c = ax.fill_between(steps_plot, 0, mean_c,
                                  color="#4CAF50", alpha=0.75, label="correct")
            h_n = ax.fill_between(steps_plot, mean_c, mean_c + mean_n,
                                  color="#FF9800", alpha=0.75, label="not_attempted (IDK)")
            h_i = ax.fill_between(steps_plot, mean_c + mean_n, mean_c + mean_n + mean_i,
                                  color="#F44336", alpha=0.75, label="incorrect")

            # Overlay line for correct fraction
            ax.plot(steps_plot, mean_c, color="#1B5E20", linewidth=2.0, linestyle="-",
                    marker="o", markersize=5)

            if legend_handles is None:
                legend_handles = [h_c, h_n, h_i]

            # Annotate final step value
            if steps_plot:
                ax.annotate(
                    f"final correct\n={mean_c[-1]:.2f}",
                    xy=(steps_plot[-1], mean_c[-1]),
                    xytext=(-50, 15),
                    textcoords="offset points",
                    fontsize=9,
                    color="#1B5E20",
                    arrowprops=dict(arrowstyle="->", color="#1B5E20"),
                )

            n = len(bin_sids)
            baseline_lbl = "step 0" if baseline_step == 0 else f"step {baseline_step}"
            ax.set_title(f"{exp_label}\n{conf_label} (n={n})", fontsize=10)
            ax.set_xlabel("Global training step", fontsize=10)
            ax.set_ylabel("Avg fraction of 32 samples", fontsize=10)
            ax.set_xticks(steps_plot)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.2)

    # Add shared legend to bottom-right
    if legend_handles is not None:
        fig.legend(
            handles=legend_handles,
            labels=["correct", "not_attempted (IDK)", "incorrect"],
            loc="lower center",
            ncol=3,
            fontsize=10,
            framealpha=0.9,
            bbox_to_anchor=(0.5, -0.02),
        )

    # Row labels on the right
    for row_idx, (_, exp_label, exp_color) in enumerate(exps_rows):
        axes[row_idx][1].annotate(
            exp_label,
            xy=(1.03, 0.5),
            xycoords="axes fraction",
            fontsize=11,
            fontweight="bold",
            color=exp_color,
            va="center",
            rotation=-90,
        )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = f"{OUT_DIR}/fig13_binary_vs_ternary_protection.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS – printed to stdout for quick reference
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_stats():
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS (val_answer set)")
    print("=" * 70)

    for exp in ["binary", "ternary_adaptive", "ternary_static"]:
        exp_base = EXPS[exp]
        trajs, meta = build_per_sample_trajectory(exp_base, "val_answer")
        if trajs is None:
            print(f"\n{exp}: val_answer data unavailable")
            continue

        degraded, non_degraded = classify_degraded(trajs, meta)
        n_total   = len(degraded) + len(non_degraded)
        baseline_step = meta["baseline_step"]

        frac_deg     = [get_baseline_frac_correct(sid, trajs, meta) for sid in degraded]
        frac_nondeg  = [get_baseline_frac_correct(sid, trajs, meta) for sid in non_degraded]
        frac_deg     = [v for v in frac_deg    if not np.isnan(v)]
        frac_nondeg  = [v for v in frac_nondeg if not np.isnan(v)]

        print(f"\n{exp} (baseline = step {baseline_step}):")
        print(f"  Questions with >=1 correct at baseline: {n_total}")
        print(f"  Degraded (→0 correct at step 480): {len(degraded)} ({100*len(degraded)/n_total:.1f}%)")
        if frac_deg:
            print(f"    Degraded — baseline conf mean={np.mean(frac_deg):.3f}, "
                  f"median={np.median(frac_deg):.3f}")
        if frac_nondeg:
            print(f"    Non-degraded — baseline conf mean={np.mean(frac_nondeg):.3f}, "
                  f"median={np.median(frac_nondeg):.3f}")

        # IDK fraction at baseline for degraded vs non-degraded
        if degraded:
            idk_fracs_deg = [trajs[sid][baseline_step]["not_attempted"] /
                             trajs[sid][baseline_step]["total"]
                             for sid in degraded if baseline_step in trajs[sid]]
            if idk_fracs_deg:
                print(f"    Degraded — baseline IDK frac mean={np.mean(idk_fracs_deg):.3f}")
        if non_degraded:
            idk_fracs_nd = [trajs[sid][baseline_step]["not_attempted"] /
                            trajs[sid][baseline_step]["total"]
                            for sid in non_degraded if baseline_step in trajs[sid]]
            if idk_fracs_nd:
                print(f"    Non-degraded — baseline IDK frac mean={np.mean(idk_fracs_nd):.3f}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Confidence-based Refusal Degradation Analysis (Figs 10–13)")
    print("=" * 70)

    print_summary_stats()

    fig10_confidence_distribution()
    fig11_baseline_breakdown()
    fig12_trajectory_by_confidence()
    fig13_binary_vs_ternary_protection()

    print()
    print("All figures saved to:", OUT_DIR)
