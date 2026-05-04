"""Analyze GRPO rollouts for entropy collapse / advantage starvation.

Per step, computes:
  - pass@1 (mean reward, mapped to 0/1)
  - frac_dead_groups: groups where all rollouts share one reward -> advantage = 0
  - mean_reward_std: avg within-group std of rewards (proxy for advantage magnitude)
  - unique_output_frac: avg fraction of unique outputs per group
  - unique_when_dead: same but restricted to dead groups (behavioral entropy with no learning signal)
  - mean_output_len / char_entropy: lexical diversity proxies

Writes CSV + plots + markdown report.
"""
import json, os, sys, math, collections
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--rollout-dir", default="/work/hdd/bbsg/twei2/rl/verl/outputs/rl/richqa_grpo/binary/rollouts")
_args, _ = _ap.parse_known_args()
ROLLOUT_DIR = Path(_args.rollout_dir)
OUT_DIR = ROLLOUT_DIR.parent
PLOT_DIR = OUT_DIR / "plots" / "entropy_collapse"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def char_entropy(s: str) -> float:
    if not s:
        return 0.0
    c = collections.Counter(s)
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def analyze_file(path: Path):
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    step = rows[0]["step"]
    # group by prompt
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["input"]].append(r)

    rewards = np.array([r["score"] for r in rows], dtype=np.float64)
    mean_r = float(rewards.mean())
    pass1 = float((rewards > 0).mean())

    dead = 0
    stds = []
    uniq_fracs = []
    uniq_when_dead = []
    out_lens = []
    out_ents = []
    for inp, grp in groups.items():
        rs = np.array([g["score"] for g in grp])
        outs = [g["output"] for g in grp]
        stds.append(rs.std())
        uf = len(set(outs)) / len(outs)
        uniq_fracs.append(uf)
        if rs.std() == 0:
            dead += 1
            uniq_when_dead.append(uf)
        out_lens.extend(len(o) for o in outs)
        out_ents.extend(char_entropy(o) for o in outs)

    return {
        "step": step,
        "n_groups": len(groups),
        "group_size": len(rows) // len(groups),
        "pass1": pass1,
        "mean_reward": mean_r,
        "frac_dead_groups": dead / len(groups),
        "mean_reward_std": float(np.mean(stds)),
        "mean_unique_output_frac": float(np.mean(uniq_fracs)),
        "mean_unique_output_frac_dead": float(np.mean(uniq_when_dead)) if uniq_when_dead else float("nan"),
        "mean_output_len": float(np.mean(out_lens)),
        "mean_char_entropy": float(np.mean(out_ents)),
    }


def main():
    files = sorted(ROLLOUT_DIR.glob("*.jsonl"), key=lambda p: int(p.stem))
    print(f"Found {len(files)} rollout files")
    records = []
    for i, f in enumerate(files):
        records.append(analyze_file(f))
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(files)}")
    df = pd.DataFrame(records).sort_values("step").reset_index(drop=True)
    csv_path = PLOT_DIR / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print("wrote", csv_path)

    # Plots
    def _plot(cols, fname, title, ylabel):
        fig, ax = plt.subplots(figsize=(9, 4))
        for c in cols:
            ax.plot(df["step"], df[c], label=c, linewidth=1)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / fname, dpi=120)
        plt.close(fig)

    _plot(["pass1"], "pass1.png", "Pass@1 over training", "pass@1")
    _plot(["frac_dead_groups"], "dead_groups.png",
          "Fraction of groups with zero advantage (all rollouts share reward)", "fraction")
    _plot(["mean_reward_std"], "reward_std.png",
          "Mean within-group reward std (advantage magnitude proxy)", "std")
    _plot(["mean_unique_output_frac", "mean_unique_output_frac_dead"],
          "unique_outputs.png",
          "Mean unique-output fraction per group (overall vs dead-only)", "unique frac")
    _plot(["mean_output_len"], "output_len.png", "Mean output length (chars)", "chars")
    _plot(["mean_char_entropy"], "char_entropy.png", "Mean char-level entropy", "bits/char")
    print("wrote plots to", PLOT_DIR)

    # Qualitative samples: dead groups at a few steps
    sample_steps = [files[0], files[len(files) // 2], files[-1]]
    samples = []
    for path in sample_steps:
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        step = rows[0]["step"]
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r["input"]].append(r)
        dead_all_wrong = []
        dead_all_right = []
        alive = []
        for inp, grp in groups.items():
            rs = [g["score"] for g in grp]
            if all(r == rs[0] for r in rs):
                if rs[0] > 0:
                    dead_all_right.append((inp, grp))
                else:
                    dead_all_wrong.append((inp, grp))
            else:
                alive.append((inp, grp))
        samples.append({
            "step": step,
            "n_dead_right": len(dead_all_right),
            "n_dead_wrong": len(dead_all_wrong),
            "n_alive": len(alive),
            "example_dead_wrong": dead_all_wrong[0] if dead_all_wrong else None,
            "example_dead_right": dead_all_right[0] if dead_all_right else None,
            "example_alive": alive[0] if alive else None,
        })

    # Markdown report
    md = [
        "# Entropy Collapse Analysis — richqa_grpo/binary",
        "",
        f"Source: `{ROLLOUT_DIR}` ({len(files)} steps, {df['n_groups'].iloc[0]} prompts × {df['group_size'].iloc[0]} rollouts per step)",
        "",
        "## Key metrics (first / mid / last)",
        "",
        "| metric | step " + f"{int(df['step'].iloc[0])}" + " | step " + f"{int(df['step'].iloc[len(df)//2])}" + " | step " + f"{int(df['step'].iloc[-1])}" + " |",
        "|---|---|---|---|",
    ]
    for col in ["pass1", "frac_dead_groups", "mean_reward_std",
                "mean_unique_output_frac", "mean_unique_output_frac_dead",
                "mean_output_len", "mean_char_entropy"]:
        a = df[col].iloc[0]
        b = df[col].iloc[len(df) // 2]
        c = df[col].iloc[-1]
        md.append(f"| {col} | {a:.4f} | {b:.4f} | {c:.4f} |")

    md += [
        "",
        "## Interpretation",
        "",
        "- **frac_dead_groups**: fraction of GRPO groups where every rollout received the same reward. "
        "Because GRPO advantages are (reward − group_mean)/group_std, these groups contribute **zero** "
        "advantage to every token. If this rises over training, the policy gradient is being fed by an "
        "ever-shrinking subset of prompts.",
        "- **mean_reward_std**: direct magnitude of the advantage signal (bounded above by 1 for ±1 rewards).",
        "- **mean_unique_output_frac**: behavioral entropy — how many distinct generations per prompt. "
        "When compared against `mean_unique_output_frac_dead`, low values in dead groups indicate the "
        "model has memorized a single output (collapse), while high values mean diverse-but-all-wrong "
        "(or all-right) generations — the loss is still blind to them.",
        "- **mean_char_entropy**: cheap lexical entropy proxy (no logprobs available in these rollouts).",
        "",
        "## Qualitative samples",
        "",
    ]
    for s in samples:
        md.append(f"### Step {s['step']}")
        md.append(f"- dead-all-correct groups: {s['n_dead_right']}")
        md.append(f"- dead-all-wrong groups: {s['n_dead_wrong']}")
        md.append(f"- alive (mixed) groups: {s['n_alive']}")
        md.append("")
        for tag, key in [("dead-all-wrong", "example_dead_wrong"),
                         ("dead-all-correct", "example_dead_right"),
                         ("alive", "example_alive")]:
            ex = s[key]
            if ex is None:
                continue
            inp, grp = ex
            md.append(f"**{tag} example (step {s['step']})**")
            md.append("")
            md.append("```")
            md.append("INPUT (truncated):")
            md.append(inp[-600:])
            md.append("")
            md.append(f"SCORES: {[g['score'] for g in grp]}")
            md.append("OUTPUTS:")
            for g in grp:
                md.append(f"  - {g['output'][:200]!r}")
            md.append("```")
            md.append("")

    md.append("## Files")
    md.append("")
    md.append(f"- Metrics CSV: `plots/entropy_collapse/metrics.csv`")
    md.append(f"- Plots: `plots/entropy_collapse/*.png`")

    report = OUT_DIR / "ANALYSIS_entropy_collapse.md"
    report.write_text("\n".join(md))
    print("wrote", report)


if __name__ == "__main__":
    main()
