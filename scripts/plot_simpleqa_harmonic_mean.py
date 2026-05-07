"""Plot SimpleQA-style harmonic-mean accuracy per experiment.

SimpleQA accuracy = harmonic mean of accuracy and accuracy_attempted:
    F = 2 * accuracy * accuracy_attempted / (accuracy + accuracy_attempted)

Usage:
    python plot_simpleqa_harmonic_mean.py <dir>

<dir> can be either:
  * An experiment directory (contains global_step_* subdirs), or
  * A project directory (contains experiment subdirs).

For each experiment, writes:
  <exp>/plots/simpleqa_harmonic_mean_by_step.png
  <exp>/plots/simpleqa_harmonic_mean_scores.json
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

STEP_RE = re.compile(r"^global_step_(\d+)$")


def is_experiment_dir(p: Path) -> bool:
    return p.is_dir() and any(STEP_RE.match(c.name) for c in p.iterdir() if c.is_dir())


def find_experiments(root: Path):
    if is_experiment_dir(root):
        return [root]
    exps = []
    for c in sorted(root.iterdir()):
        if c.is_dir() and is_experiment_dir(c):
            exps.append(c)
    return exps


def harmonic_mean(acc, acc_attempted):
    if acc is None or acc_attempted is None:
        return None
    s = acc + acc_attempted
    if s <= 0:
        return 0.0
    return 2 * acc * acc_attempted / s


def collect(exp_dir: Path):
    """Return {split: [(step, f, acc, acc_attempted), ...]}, sorted by step."""
    out = {}
    for ckpt in exp_dir.iterdir():
        if not ckpt.is_dir():
            continue
        m = STEP_RE.match(ckpt.name)
        if not m:
            continue
        step = int(m.group(1))
        gen_dir = ckpt / "generations"
        if not gen_dir.is_dir():
            continue
        for j in gen_dir.glob("*_eval.json"):
            mm = re.search(r"__on_(.+)_eval\.json$", j.name)
            if not mm:
                continue
            split = mm.group(1)
            try:
                metrics = json.loads(j.read_text()).get("metrics", {})
            except Exception as e:
                print(f"  skip {j}: {e}")
                continue
            acc = metrics.get("accuracy")
            acc_a = metrics.get("accuracy_attempted")
            f = harmonic_mean(acc, acc_a)
            if f is None:
                continue
            out.setdefault(split, []).append((step, f, acc, acc_a))
    for k in out:
        out[k].sort()
    return out


def process_experiment(exp_dir: Path):
    data = collect(exp_dir)
    if not data:
        print(f"  no eval data in {exp_dir}")
        return
    out_dir = exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = {
        split: [
            {"step": s, "simpleqa_f1": f, "accuracy": a, "accuracy_attempted": aa}
            for (s, f, a, aa) in rows
        ]
        for split, rows in data.items()
    }
    (out_dir / "simpleqa_harmonic_mean_scores.json").write_text(
        json.dumps(scores, indent=2)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for i, (split, rows) in enumerate(sorted(data.items())):
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        ax.plot(xs, ys, marker="o", color=cmap(i % 10), label=split)
    ax.set_xlabel("Global step")
    ax.set_ylabel("SimpleQA accuracy (harmonic mean)")
    ax.set_title(f"{exp_dir.name}: SimpleQA harmonic-mean accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "simpleqa_harmonic_mean_by_step.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_dir}/simpleqa_harmonic_mean_by_step.png  (splits: {sorted(data)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path, help="experiment dir or project dir")
    args = ap.parse_args()

    root = args.dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    exps = find_experiments(root)
    if not exps:
        raise SystemExit(f"no experiments with global_step_* checkpoints under {root}")

    print(f"Found {len(exps)} experiment(s) under {root}")
    for exp in exps:
        print(f"- {exp.name}")
        process_experiment(exp)


if __name__ == "__main__":
    main()
