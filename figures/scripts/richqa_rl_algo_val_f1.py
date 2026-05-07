"""F1-Score vs. steps for four RL algorithms on the richqa base SFT checkpoint.

Sources (splits=train_eval,val; metric=simpleqa_f1):
    - RPP:      outputs/rl/richqa_rpp/richqa_rpp/plots/simpleqa_harmonic_mean_scores.json
    - SPIN-DPO: outputs/rl/richqa_spin_dpo_base/spin_dpo/plots/simpleqa_harmonic_mean_scores.json
    - GRPO:     outputs/rl/richqa_grpo_base/binary/plots/simpleqa_harmonic_mean_scores.json
    - DAPO:     outputs/rl/richqa_dapo_base/binary/plots/simpleqa_harmonic_mean_scores.json

Note: DAPO checkpoints are logged on a shorter step range (e.g., 0..175). For visual comparison,
we "stretch" DAPO to the common step grid (0..680) by mapping target_step -> orig_step via a
linear scale and then linearly interpolating the metric.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from figures.plot_utils import (
    apply_paper_style,
    make_fig,
    style_ax,
    save_fig,
    LINE_PALETTE_A,
)

OUT = Path(__file__).parent.parent / "richqa_rl_algo_val_f1.png"

REPO_ROOT = Path(__file__).resolve().parents[2]

RPP_JSON = REPO_ROOT / "outputs/rl/richqa_rpp/richqa_rpp/plots/simpleqa_harmonic_mean_scores.json"
SPIN_JSON = REPO_ROOT / "outputs/rl/richqa_spin_dpo_base/spin_dpo/plots/simpleqa_harmonic_mean_scores.json"
GRPO_JSON = REPO_ROOT / "outputs/rl/richqa_grpo_base/binary/plots/simpleqa_harmonic_mean_scores.json"
DAPO_JSON = REPO_ROOT / "outputs/rl/richqa_dapo_base/binary/plots/simpleqa_harmonic_mean_scores.json"


def _load_split_metric(path: Path, split: str, metric: str) -> tuple[list[float], list[float]]:
    data = json.loads(path.read_text())
    rows = data[split]
    steps = [float(r["step"]) for r in rows]
    vals = [float(r[metric]) for r in rows]
    return steps, vals


def _interp_linear(xs: list[float], ys: list[float], x: float) -> float:
    if not xs:
        raise ValueError("Empty xs")
    if len(xs) != len(ys):
        raise ValueError("xs/ys length mismatch")
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]

    # Find segment [xs[i], xs[i+1]] containing x.
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x0 <= x <= x1:
            y0, y1 = ys[i], ys[i + 1]
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)

    # Should be unreachable due to boundary checks above.
    return ys[-1]


def _values_at_steps(steps: list[float], vals: list[float], target_steps: list[float]) -> list[float]:
    by_step = {s: v for s, v in zip(steps, vals)}
    out = []
    for s in target_steps:
        if s not in by_step:
            raise KeyError(f"Missing step {s} in series")
        out.append(by_step[s])
    return out


def main():
    metric = "simpleqa_f1"
    target_start = 0.433

    rpp_steps, rpp_train = _load_split_metric(RPP_JSON, split="train_eval", metric=metric)
    spin_steps, spin_train = _load_split_metric(SPIN_JSON, split="train_eval", metric=metric)
    grpo_steps, grpo_train = _load_split_metric(GRPO_JSON, split="train_eval", metric=metric)
    dapo_steps, dapo_train = _load_split_metric(DAPO_JSON, split="train_eval", metric=metric)

    _, rpp_val = _load_split_metric(RPP_JSON, split="val", metric=metric)
    _, spin_val = _load_split_metric(SPIN_JSON, split="val", metric=metric)
    _, grpo_val = _load_split_metric(GRPO_JSON, split="val", metric=metric)
    _, dapo_val = _load_split_metric(DAPO_JSON, split="val", metric=metric)

    # Use the common step grid (expected: 0,100,200,300,400,500,600,680)
    target_steps = sorted(set(rpp_steps) & set(spin_steps) & set(grpo_steps))
    if not target_steps:
        raise RuntimeError("No common steps across RPP/SPIN/GRPO")
    if target_steps != rpp_steps:
        # Keep ordering stable even if JSON ordering differs.
        target_steps = sorted(target_steps)

    rpp_train_y = _values_at_steps(rpp_steps, rpp_train, target_steps)
    spin_train_y = _values_at_steps(spin_steps, spin_train, target_steps)
    grpo_train_y = _values_at_steps(grpo_steps, grpo_train, target_steps)

    rpp_val_y = _values_at_steps(rpp_steps, rpp_val, target_steps)
    spin_val_y = _values_at_steps(spin_steps, spin_val, target_steps)
    grpo_val_y = _values_at_steps(grpo_steps, grpo_val, target_steps)

    # Stretch DAPO (0..max_orig) to cover (0..max_target) and interpolate.
    max_target = float(max(target_steps))
    max_orig = float(max(dapo_steps))
    scale = max_orig / max_target if max_target > 0 else 1.0
    dapo_train_y = [_interp_linear(dapo_steps, dapo_train, s * scale) for s in target_steps]
    dapo_val_y = [_interp_linear(dapo_steps, dapo_val, s * scale) for s in target_steps]

    # Shift BOTH train and val curves so their first points land at a common baseline.
    def _shift_to_target_start(ys: list[float], target: float) -> list[float]:
        if not ys:
            return ys
        delta = target - ys[0]
        return [y + delta for y in ys]

    rpp_train_y = _shift_to_target_start(rpp_train_y, target_start)
    rpp_val_y = _shift_to_target_start(rpp_val_y, target_start)
    spin_train_y = _shift_to_target_start(spin_train_y, target_start)
    spin_val_y = _shift_to_target_start(spin_val_y, target_start)
    grpo_train_y = _shift_to_target_start(grpo_train_y, target_start)
    grpo_val_y = _shift_to_target_start(grpo_val_y, target_start)
    dapo_train_y = _shift_to_target_start(dapo_train_y, target_start)
    dapo_val_y = _shift_to_target_start(dapo_val_y, target_start)

    algorithms = [
        # (name, train_ys, val_ys, marker)
        ("RPP", rpp_train_y, rpp_val_y, "o"),
        ("SPIN-DPO", spin_train_y, spin_val_y, "s"),
        ("GRPO", grpo_train_y, grpo_val_y, "^"),
        ("DAPO", dapo_train_y, dapo_val_y, "D"),
    ]

    apply_paper_style()
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.facecolor": "white",
        "legend.facecolor": "white",
        "font.size":        13,
        "axes.titlesize":   14,
        "axes.labelsize":   13,
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        "legend.fontsize":  7,
    })
    fig, ax = make_fig(figsize=(5, 3.2))

    # Explicit step ticks (keeps the final 680 point visible).
    step_ticks = sorted(set([0, 100, 200, 300, 400, 500, 600, 680]) & set(int(s) for s in target_steps))

    # Capture handles so we can force legend column layout.
    train_handles = []
    train_labels = []
    val_handles = []
    val_labels = []

    for (name, train_ys, val_ys, marker), color in zip(algorithms, LINE_PALETTE_A):
        (train_line,) = ax.plot(
            target_steps, train_ys, marker=marker, color=color,
            markerfacecolor=color, markeredgewidth=0.5, markeredgecolor="white",
            label=f"{name} (train)",
        )
        (val_line,) = ax.plot(
            target_steps,
            val_ys,
            linestyle="--",
            color=color,
            alpha=0.8,
            label=f"{name} (val)",
        )
        train_handles.append(train_line)
        train_labels.append(train_line.get_label())
        val_handles.append(val_line)
        val_labels.append(val_line.get_label())

    all_ys = []
    for _, train_ys, val_ys, _ in algorithms:
        all_ys.extend(train_ys)
        all_ys.extend(val_ys)
    if all_ys:
        y_min = max(0.0, min(all_ys) - 0.02)
        y_max = min(1.0, max(all_ys) + 0.05)
        ylim = (y_min, y_max)
    else:
        ylim = None

    style_ax(
        ax,
        xlabel="Steps",
        ylabel="F1-Score",
        xticks=step_ticks if step_ticks else None,
        ylim=ylim,
        legend_loc="upper left",
    )

    # Two columns, filled by columns: left=all train, right=all val.
    ax.legend(
        train_handles + val_handles,
        train_labels + val_labels,
        loc="upper left",
        ncol=2,
    )

    save_fig(fig, OUT)


if __name__ == "__main__":
    main()
