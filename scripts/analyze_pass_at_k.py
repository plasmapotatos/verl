#!/usr/bin/env python3
"""
Analyze pass@k data to determine knowledge splits for SimpleQA.

Generates:
  - outputs/simpleqa_qwen_pass_at_k/analysis/pass_at_k_overview.png
  - outputs/simpleqa_qwen_pass_at_k/analysis/correct_count_distribution.png
  - outputs/simpleqa_qwen_pass_at_k/analysis/binomial_test.png
  - outputs/simpleqa_qwen_pass_at_k/analysis/split_summary.txt
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import binom


def load_eval(output_root: Path, k: int) -> list[dict]:
    p = output_root / f"k_{k}" / "pass@k" / f"eval_{k}.json"
    data = json.loads(p.read_text())
    return data["rows"]


def per_question_correct_counts(rows: list[dict]) -> np.ndarray:
    """Return array of shape (n_questions,) with # correct rollouts per question."""
    counts = []
    for r in rows:
        n_correct = sum(1 for g in r["graders"] if g["evaluation"] == "correct")
        counts.append(n_correct)
    return np.array(counts)


def load_pass_at_k_summary(output_root: Path, k: int) -> float:
    p = output_root / f"k_{k}" / "pass@k" / f"pass_at_k_{k}.json"
    data = json.loads(p.read_text())
    return data["pass_at_k"]["pass_at_k"]


def binomial_pvalue_per_question(counts: np.ndarray, k: int, null_p: float) -> np.ndarray:
    """One-sided binomial test: P(X >= observed | H0: p = null_p)."""
    pvals = np.array([binom.sf(c - 1, k, null_p) for c in counts])
    return pvals


def fdr_benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """BH correction. Returns boolean mask of rejected nulls."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked_pvals = pvals[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    reject = ranked_pvals <= thresholds
    # find largest k that satisfies threshold
    if not np.any(reject):
        return np.zeros(n, dtype=bool)
    max_k = np.where(reject)[0].max()
    result = np.zeros(n, dtype=bool)
    result[order[: max_k + 1]] = True
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="outputs/simpleqa_qwen_pass_at_k",
        help="Root directory of pass@k outputs",
    )
    parser.add_argument(
        "--available-k",
        nargs="+",
        type=int,
        default=[1, 16, 32, 64],
        help="Which k values have completed runs",
    )
    parser.add_argument(
        "--primary-k",
        type=int,
        default=64,
        help="K to use for per-question split analysis",
    )
    parser.add_argument(
        "--null-p",
        type=float,
        default=0.003,
        help="Baseline per-rollout correct probability under null (lucky guess rate)",
    )
    parser.add_argument(
        "--fdr-alpha",
        type=float,
        default=0.05,
        help="FDR threshold for binomial significance test",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # ── 1. pass@k curve ────────────────────────────────────────────────────────
    k_vals = args.available_k
    pass_rates = [load_pass_at_k_summary(output_root, k) for k in k_vals]

    # ── 2. per-question distribution at primary k ──────────────────────────────
    rows = load_eval(output_root, args.primary_k)
    counts = per_question_correct_counts(rows)
    n_questions = len(counts)
    fractions = counts / args.primary_k

    # ── 3. binomial significance ───────────────────────────────────────────────
    pvals = binomial_pvalue_per_question(counts, args.primary_k, args.null_p)
    significant = fdr_benjamini_hochberg(pvals, args.fdr_alpha)

    # ── 4. define splits ───────────────────────────────────────────────────────
    # Tiered: "knows" / "learnable" / "unknown"
    # Knows: fraction >= 0.25 (≥16/64) — reliably correct
    # Learnable: 0 < fraction < 0.25   — has signal but unreliable
    # Unknown: fraction == 0           — no correct rollouts
    thresholds = {
        "knows_strict":    fractions >= 0.5,
        "knows":           fractions >= 0.25,
        "learnable_any":   (fractions > 0) & (fractions < 0.25),
        "learnable_2plus": (counts >= 2) & (fractions < 0.25),
        "learnable_4plus": (counts >= 4) & (fractions < 0.25),
        "unknown":         counts == 0,
        "binomial_sig":    significant,
    }

    # ── Figure 1: overview (2x2) ───────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Panel A: pass@k curve
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(k_vals, [r * 100 for r in pass_rates], marker="o", color="#1d4ed8", linewidth=2)
    ax_a.set_title("A. pass@k (any correct in k rollouts)", fontweight="bold")
    ax_a.set_xlabel("k")
    ax_a.set_ylabel("pass@k (%)")
    ax_a.grid(True, axis="y", linestyle="--", alpha=0.5)
    for x, y in zip(k_vals, pass_rates):
        ax_a.annotate(f"{y*100:.1f}%", (x, y * 100), textcoords="offset points",
                      xytext=(0, 7), ha="center", fontsize=8)

    # Panel B: histogram of correct_count at primary k (log scale)
    ax_b = fig.add_subplot(gs[0, 1])
    max_count = counts.max()
    bins = np.arange(-0.5, max_count + 1.5, 1)
    ax_b.hist(counts, bins=bins, color="#6366f1", edgecolor="white", linewidth=0.3)
    ax_b.set_yscale("log")
    ax_b.set_title(f"B. # correct rollouts per question (k={args.primary_k})", fontweight="bold")
    ax_b.set_xlabel(f"# correct out of {args.primary_k}")
    ax_b.set_ylabel("# questions (log scale)")
    ax_b.axvline(1.5, color="#ef4444", linestyle="--", linewidth=1.5, label="≥2 cutoff")
    ax_b.axvline(3.5, color="#f97316", linestyle="--", linewidth=1.5, label="≥4 cutoff")
    ax_b.axvline(int(args.primary_k * 0.25) - 0.5, color="#22c55e", linestyle="--",
                 linewidth=1.5, label=f"≥25% ({int(args.primary_k*0.25)}) cutoff")
    ax_b.legend(fontsize=7)

    # Panel C: zoomed histogram (1-20 correct, excluding 0)
    ax_c = fig.add_subplot(gs[1, 0])
    nonzero = counts[counts > 0]
    zoom_bins = np.arange(0.5, min(nonzero.max() + 1.5, 30), 1)
    ax_c.hist(nonzero, bins=zoom_bins, color="#8b5cf6", edgecolor="white", linewidth=0.3)
    ax_c.set_title(f"C. Zoomed: questions with ≥1 correct (k={args.primary_k})", fontweight="bold")
    ax_c.set_xlabel(f"# correct out of {args.primary_k}")
    ax_c.set_ylabel("# questions")
    ax_c.axvline(1.5, color="#ef4444", linestyle="--", linewidth=1.5, label="≥2")
    ax_c.axvline(3.5, color="#f97316", linestyle="--", linewidth=1.5, label="≥4")
    ax_c.axvline(int(args.primary_k * 0.25) - 0.5, color="#22c55e", linestyle="--",
                 linewidth=1.5, label=f"≥25%")
    ax_c.legend(fontsize=7)
    ax_c.set_xlim(0.5, min(nonzero.max() + 1, 35))

    # Panel D: cumulative % of questions vs fraction threshold
    ax_d = fig.add_subplot(gs[1, 1])
    frac_thresholds = np.linspace(0, 1, 200)
    pct_above = [100 * np.mean(fractions >= t) for t in frac_thresholds]
    ax_d.plot(frac_thresholds * 100, pct_above, color="#0891b2", linewidth=2)
    ax_d.set_title(f"D. % questions above correct-rate threshold (k={args.primary_k})",
                   fontweight="bold")
    ax_d.set_xlabel("Threshold: correct_count / k  (%)")
    ax_d.set_ylabel("% questions above threshold")
    ax_d.grid(True, linestyle="--", alpha=0.4)
    # mark key thresholds
    for thresh, label, color in [
        (1/args.primary_k, f"≥1/{args.primary_k}", "#94a3b8"),
        (2/args.primary_k, f"≥2/{args.primary_k}", "#ef4444"),
        (4/args.primary_k, f"≥4/{args.primary_k}", "#f97316"),
        (0.25, "≥25%", "#22c55e"),
        (0.5, "≥50%", "#1d4ed8"),
    ]:
        y = 100 * np.mean(fractions >= thresh)
        ax_d.axvline(thresh * 100, color=color, linestyle=":", linewidth=1.2)
        ax_d.annotate(f"{label}\n({y:.1f}%)", (thresh * 100, y),
                      textcoords="offset points", xytext=(4, -10),
                      fontsize=7, color=color)

    fig.suptitle(
        f"Qwen-2.5-3B SimpleQA: pass@k analysis  (n={n_questions:,} questions)",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(analysis_dir / "pass_at_k_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {analysis_dir / 'pass_at_k_overview.png'}")

    # ── Figure 2: binomial test visualization ─────────────────────────────────
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1, ax2 = axes
    log_pvals = -np.log10(pvals + 1e-300)
    ax1.hist(log_pvals, bins=80, color="#6366f1", edgecolor="white", linewidth=0.2)
    ax1.set_xlabel(r"$-\log_{10}(p\text{-value})$")
    ax1.set_ylabel("# questions")
    ax1.set_title(
        f"Binomial test p-values (null p={args.null_p}, k={args.primary_k})\n"
        f"(one-sided: does question exceed baseline guessing rate?)",
        fontweight="bold",
    )
    n_sig = significant.sum()
    ax1.axvline(-np.log10(args.fdr_alpha), color="#ef4444", linestyle="--",
                label=f"raw α={args.fdr_alpha} ({(-np.log10(args.fdr_alpha)):.1f})")
    ax1.legend(fontsize=8)
    ax1.annotate(f"BH-significant: {n_sig} questions\n({n_sig/n_questions*100:.1f}%)",
                 xy=(0.62, 0.75), xycoords="axes fraction", fontsize=9,
                 bbox=dict(boxstyle="round", fc="lightyellow"))

    # scatter: correct_count vs -log10(pval)
    jitter = np.random.default_rng(0).uniform(-0.3, 0.3, n_questions)
    ax2.scatter(counts + jitter, log_pvals, alpha=0.15, s=4, c="#6366f1")
    ax2.scatter(counts[significant] + jitter[significant], log_pvals[significant],
                alpha=0.5, s=8, c="#ef4444", label=f"BH-sig (n={n_sig})")
    ax2.set_xlabel(f"# correct rollouts out of {args.primary_k}")
    ax2.set_ylabel(r"$-\log_{10}(p\text{-value})$")
    ax2.set_title("Correct count vs significance", fontweight="bold")
    ax2.legend(fontsize=8)

    fig2.suptitle(f"Binomial significance (null_p={args.null_p}, FDR α={args.fdr_alpha})",
                  fontsize=12, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(analysis_dir / "binomial_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {analysis_dir / 'binomial_test.png'}")

    # ── Text summary ───────────────────────────────────────────────────────────
    lines = [
        "=" * 60,
        "PASS@K ANALYSIS SUMMARY",
        "=" * 60,
        "",
        "pass@k rates:",
        *[f"  k={k:4d}: {r*100:.2f}%" for k, r in zip(k_vals, pass_rates)],
        "",
        f"Per-question analysis at k={args.primary_k} (n={n_questions:,}):",
        "",
        "  correct=0  (unknown):        "
        f"{(counts==0).sum():5d} ({(counts==0).mean()*100:.1f}%)",
        "  correct=1  (single hit):     "
        f"{(counts==1).sum():5d} ({(counts==1).mean()*100:.1f}%)",
        "  correct>=2 (≥3.1% rate):     "
        f"{(counts>=2).sum():5d} ({(counts>=2).mean()*100:.1f}%)",
        "  correct>=4 (≥6.3% rate):     "
        f"{(counts>=4).sum():5d} ({(counts>=4).mean()*100:.1f}%)",
        f"  correct>={int(args.primary_k*0.1):2d} (≥10% rate):      "
        f"{(counts>=int(args.primary_k*0.1)).sum():5d} "
        f"({(counts>=int(args.primary_k*0.1)).mean()*100:.1f}%)",
        f"  correct>={int(args.primary_k*0.25):2d} (≥25% rate):      "
        f"{(counts>=int(args.primary_k*0.25)).sum():5d} "
        f"({(counts>=int(args.primary_k*0.25)).mean()*100:.1f}%)",
        f"  correct>={int(args.primary_k*0.5):2d} (≥50% rate):      "
        f"{(counts>=int(args.primary_k*0.5)).sum():5d} "
        f"({(counts>=int(args.primary_k*0.5)).mean()*100:.1f}%)",
        "",
        f"Binomial test (null_p={args.null_p}, FDR α={args.fdr_alpha}):",
        f"  Significant questions: {n_sig} ({n_sig/n_questions*100:.1f}%)",
        "",
        "RECOMMENDED SPLITS:",
        "",
        "  Tier 1 — 'Knows' (≥25% correct rate at k=64):",
        f"    {thresholds['knows'].sum()} questions ({thresholds['knows'].mean()*100:.1f}%)",
        "    → Reliable knowledge; use as held-out eval or keep-alive set",
        "",
        "  Tier 2 — 'Learnable' (1-15 correct / 64, i.e. 1-24%):",
        f"    {thresholds['learnable_any'].sum()} questions ({thresholds['learnable_any'].mean()*100:.1f}%)",
        "    → Has signal but unreliable; IDEAL RL training targets",
        "    → Subfilter ≥2 correct (cleaner signal):",
        f"      {thresholds['learnable_2plus'].sum()} questions ({thresholds['learnable_2plus'].mean()*100:.1f}%)",
        "    → Subfilter ≥4 correct (conservative):",
        f"      {thresholds['learnable_4plus'].sum()} questions ({thresholds['learnable_4plus'].mean()*100:.1f}%)",
        "",
        "  Tier 3 — 'Unknown' (0 correct / 64):",
        f"    {thresholds['unknown'].sum()} questions ({thresholds['unknown'].mean()*100:.1f}%)",
        "    → No signal; model can't learn from RL on these",
        "",
        "  Binomial-significant (statistically above null):",
        f"    {n_sig} questions ({n_sig/n_questions*100:.1f}%)",
        "    → Most principled 'knows' boundary",
        "=" * 60,
    ]
    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    (analysis_dir / "split_summary.txt").write_text(summary_text)
    print(f"\nSaved: {analysis_dir / 'split_summary.txt'}")


if __name__ == "__main__":
    main()
