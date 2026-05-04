"""Plot accuracy curves across multiple RL experiments.

Reads metrics.accuracy from each checkpoint's generations/*_on_{val,train_eval}_eval.json
and plots val-only, train_eval-only, and combined figures.

Also writes a short train-improvement analysis.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

EXPERIMENTS = [
    ("factual_anchor/binary", "/work/hdd/bbsg/twei2/rl/verl/outputs/rl/factual_anchor_grpo_unbracketed_question/binary_unbracketed_question"),
    ("factual_anchor/bag_a0.5b1", "/work/hdd/bbsg/twei2/rl/verl/outputs/rl/factual_anchor_grpo_unbracketed_question/correct_plus_novel_bag_a0.5b1_unbracketed_question"),
    ("factual_anchor/bag_a1b0.5", "/work/hdd/bbsg/twei2/rl/verl/outputs/rl/factual_anchor_grpo_unbracketed_question/correct_plus_novel_bag_a1b0.5_unbracketed_question"),
    ("factual_anchor/bag_alpha0.5", "/work/hdd/bbsg/twei2/rl/verl/outputs/rl/factual_anchor_grpo_unbracketed_question/correct_plus_novel_bag_alpha0.5_unbracketed_question"),
    ("factual_anchor/bag_alpha1.0", "/work/hdd/bbsg/twei2/rl/verl/outputs/rl/factual_anchor_grpo_unbracketed_question/correct_plus_novel_bag_alpha1.0_unbracketed_question"),
    ("richqa_dapo_base", "/work/hdd/bbsg/twei2/rl/verl/outputs/rl/richqa_dapo_base/binary"),
    ("richqa_grpo_base", "/work/hdd/bbsg/twei2/rl/verl/outputs/rl/richqa_grpo_base/binary"),
    ("richqa_rpp", "/work/hdd/bbsg/twei2/rl/verl/outputs/rl/richqa_rpp/richqa_rpp"),
    ("rich_qa_paraphrased_smoketest", "/work/hdd/bbsg/twei2/rl/verl/outputs/rl/simpleqa_rich_qa_paraphrased_smoketest_grpo/binary_rich_qa_paraphrased_smoketest"),
    ("rich_qa_paraphrased", "/work/hdd/bbsg/twei2/rl/verl/outputs/rl/simpleqa_rich_qa_paraphrased_grpo/binary_rich_qa_paraphrased"),
]

OUT_DIR = Path("/work/hdd/bbsg/twei2/rl/verl/outputs/rl/_cross_experiment_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect(exp_dir: Path):
    """Return dict: {split: [(step, acc), ...]} sorted by step."""
    out = {"val": [], "train_eval": []}
    step_re = re.compile(r"global_step_(\d+)$")
    for ckpt in exp_dir.iterdir():
        if not ckpt.is_dir():
            continue
        m = step_re.match(ckpt.name)
        if not m:
            continue
        step = int(m.group(1))
        gen_dir = ckpt / "generations"
        if not gen_dir.is_dir():
            continue
        for split in ("val", "train_eval"):
            suffix = f"__on_{split}_eval.json"
            matches = [p for p in gen_dir.iterdir() if p.name.endswith(suffix)]
            if not matches:
                continue
            try:
                data = json.loads(matches[0].read_text())
                acc = data.get("metrics", {}).get("accuracy")
                if acc is not None:
                    out[split].append((step, acc))
            except Exception as e:
                print(f"  skip {matches[0]}: {e}")
    for k in out:
        out[k].sort()
    return out


def main():
    all_data = {}
    for name, path in EXPERIMENTS:
        p = Path(path)
        if not p.is_dir():
            print(f"MISSING: {path}")
            continue
        all_data[name] = collect(p)
        n_val = len(all_data[name]["val"])
        n_tr = len(all_data[name]["train_eval"])
        print(f"{name}: val={n_val} train_eval={n_tr}")

    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i % 10) for i, name in enumerate(all_data)}

    # --- val only ---
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, d in all_data.items():
        if d["val"]:
            xs, ys = zip(*d["val"])
            ax.plot(xs, ys, marker="o", label=name, color=colors[name])
    ax.set_xlabel("Global step"); ax.set_ylabel("Accuracy")
    ax.set_title("Validation accuracy across RL experiments")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "val_accuracy.png", dpi=150)
    plt.close(fig)

    # --- train_eval only ---
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, d in all_data.items():
        if d["train_eval"]:
            xs, ys = zip(*d["train_eval"])
            ax.plot(xs, ys, marker="o", label=name, color=colors[name])
    ax.set_xlabel("Global step"); ax.set_ylabel("Accuracy")
    ax.set_title("Train-eval accuracy across RL experiments")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "train_eval_accuracy.png", dpi=150)
    plt.close(fig)

    # --- combined ---
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, d in all_data.items():
        c = colors[name]
        if d["val"]:
            xs, ys = zip(*d["val"])
            ax.plot(xs, ys, marker="o", linestyle="-", color=c, label=f"{name} (val)")
        if d["train_eval"]:
            xs, ys = zip(*d["train_eval"])
            ax.plot(xs, ys, marker="x", linestyle="--", color=c, label=f"{name} (train_eval)")
    ax.set_xlabel("Global step"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy across RL experiments (solid=val, dashed=train_eval)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "combined_accuracy.png", dpi=150)
    plt.close(fig)

    # --- train improvement analysis ---
    lines = ["# Train-eval improvement across RL experiments", ""]
    lines.append("| Experiment | step0 train | best train | Δ train | step0 val | best val | Δ val | train−val gap (best) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    rows = []
    for name, d in all_data.items():
        tr = d["train_eval"]; vl = d["val"]
        if not tr:
            continue
        tr0 = tr[0][1]; tr_best = max(y for _, y in tr); tr_best_step = max(tr, key=lambda t: t[1])[0]
        if vl:
            vl0 = vl[0][1]; vl_best = max(y for _, y in vl)
        else:
            vl0 = vl_best = float("nan")
        d_tr = tr_best - tr0
        d_vl = vl_best - vl0 if vl else float("nan")
        gap = tr_best - vl_best if vl else float("nan")
        rows.append((name, tr0, tr_best, tr_best_step, d_tr, vl0, vl_best, d_vl, gap))
        lines.append(
            f"| {name} | {tr0:.3f} | {tr_best:.3f} (step {tr_best_step}) | {d_tr:+.3f} | "
            f"{vl0:.3f} | {vl_best:.3f} | {d_vl:+.3f} | {gap:+.3f} |"
        )
    # sort by Δ train descending for commentary
    rows.sort(key=lambda r: r[4], reverse=True)
    lines += ["", "## Ranked by train-set improvement (Δ train = best − step0)", ""]
    for r in rows:
        lines.append(f"- **{r[0]}**: Δtrain={r[4]:+.3f}, Δval={r[7]:+.3f}, overfit gap (train_best − val_best)={r[8]:+.3f}")

    (OUT_DIR / "train_improvement_analysis.md").write_text("\n".join(lines))
    print(f"\nWrote plots + analysis to {OUT_DIR}")


if __name__ == "__main__":
    main()
