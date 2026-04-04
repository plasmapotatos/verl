"""Summary figure: the core finding at a glance."""

import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = '/work/hdd/bbsg/twei2/rl/verl/outputs/rl/simpleqa_rich_sft_grpo_refusal'
STEPS = [0, 100, 200, 300, 400, 480]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_eval(exp_base, step, folder):
    p = f'{exp_base}/global_step_{step}/pass@k/{folder}/eval.json'
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

def rs(row):
    evals = [g['evaluation'] for g in row['graders']]
    return dict(correct=evals.count('correct'), not_attempted=evals.count('not_attempted'),
                incorrect=evals.count('incorrect'), total=len(evals))

EXPS = {
    "binary":           f"{BASE}/simpleqa_rich_sft_grpo_refusal_binary",
    "ternary_adaptive": f"{BASE}/simpleqa_rich_sft_grpo_refusal_ternary_adaptive",
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "Refusal generalization: binary vs ternary_adaptive reward\n"
    "VAL set = questions seen only during SFT (not GRPO training)",
    fontsize=14, fontweight="bold"
)

for col, (exp_name, exp_base) in enumerate(EXPS.items()):
    # --- Panel A: pass@k over steps (VAL vs TRAIN) ---
    ax = axes[0][col]
    for split_folder, label, color, ls in [
        ('train_richqa_answer_origqa_frac0.2',       'VAL answer set',   '#E53935', '--'),
        ('train_richqa_answer_origqa_frac0.8_frac0.2','TRAIN answer set', '#1565C0', '-'),
        ('train_richqa_refusal_origqa_frac0.2',       'VAL refusal set',  '#F06292', ':'),
        ('train_richqa_refusal_origqa_frac0.8_frac0.2','TRAIN refusal set','#42A5F5',  '-.'),
    ]:
        xs, ys = [], []
        for step in STEPS:
            p = f'{exp_base}/global_step_{step}/pass@k/{split_folder}/pass_at_k.json'
            if not os.path.exists(p): continue
            with open(p) as f: d = json.load(f)
            xs.append(step); ys.append(d['pass_at_k']['pass_at_k'])
        if xs:
            ax.plot(xs, ys, color=color, linestyle=ls, marker='o', linewidth=2,
                    markersize=5, label=label)
    ax.set_title(f'{exp_name}\npass@k over steps', fontsize=11)
    ax.set_xlabel('global step'); ax.set_ylabel('pass@k')
    ax.set_ylim(0.0, 0.7); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # --- Panel B: fate of originally-correct VAL questions (stacked bar) ---
    ax = axes[1][col]
    d0   = load_eval(exp_base, 0,   'train_richqa_answer_origqa_frac0.2')
    rows0 = {r['id']: rs(r) for r in d0['rows']}
    had_correct = [sid for sid, r in rows0.items() if r['correct'] > 0]

    n_full_deg, n_partial_deg, n_stable, n_improved = [], [], [], []
    for step in STEPS:
        dS = load_eval(exp_base, step, 'train_richqa_answer_origqa_frac0.2')
        rowsS = {r['id']: rs(r) for r in dS['rows']}
        full_deg = sum(1 for sid in had_correct if rowsS[sid]['correct'] == 0)
        partial   = sum(1 for sid in had_correct if 0 < rowsS[sid]['correct'] < rows0[sid]['correct'])
        same      = sum(1 for sid in had_correct if rowsS[sid]['correct'] == rows0[sid]['correct'])
        improved  = sum(1 for sid in had_correct if rowsS[sid]['correct'] > rows0[sid]['correct'])
        n_full_deg.append(full_deg); n_partial_deg.append(partial)
        n_stable.append(same); n_improved.append(improved)

    xs = list(range(len(STEPS)))
    ax.bar(xs, n_improved,   label='improved',          color='#00796B', alpha=0.85)
    ax.bar(xs, n_stable,     label='unchanged',          color='#4CAF50', alpha=0.85, bottom=n_improved)
    bot2 = [a+b for a,b in zip(n_improved, n_stable)]
    ax.bar(xs, n_partial_deg,label='partial degradation',color='#FF9800', alpha=0.85, bottom=bot2)
    bot3 = [a+b for a,b in zip(bot2, n_partial_deg)]
    ax.bar(xs, n_full_deg,   label='fully IDK (0 correct)',color='#E53935', alpha=0.85, bottom=bot3)
    ax.set_xticks(xs); ax.set_xticklabels([f's{s}' for s in STEPS])
    ax.set_title(f'{exp_name}\nFate of VAL Qs with ≥1 correct at step 0 (n={len(had_correct)})', fontsize=10)
    ax.set_ylabel('# questions'); ax.legend(fontsize=8)

# --- Panel C: Mean trajectory of all-correct-at-0 questions, both exps ---
ax = axes[0][2]
for exp_name, exp_base in EXPS.items():
    color = '#E53935' if 'ternary' in exp_name else '#1565C0'
    d0 = load_eval(exp_base, 0, 'train_richqa_answer_origqa_frac0.2')
    rows0 = {r['id']: rs(r) for r in d0['rows']}
    all_correct_0 = [sid for sid, r in rows0.items() if r['correct'] == r['total']]

    mean_c, std_c = [], []
    for step in STEPS:
        dS = load_eval(exp_base, step, 'train_richqa_answer_origqa_frac0.2')
        rowsS = {r['id']: rs(r) for r in dS['rows']}
        vals = [rowsS[sid]['correct'] / rowsS[sid]['total'] for sid in all_correct_0]
        mean_c.append(np.mean(vals)); std_c.append(np.std(vals))
    mean_c = np.array(mean_c); std_c = np.array(std_c)
    ax.plot(STEPS, mean_c, color=color, marker='o', linewidth=2.5, label=exp_name)
    ax.fill_between(STEPS, mean_c - std_c, mean_c + std_c, color=color, alpha=0.15)

ax.set_title('Mean correct fraction for all-correct-at-0 VAL Qs\n(shading = ±1 std)', fontsize=10)
ax.set_xlabel('global step'); ax.set_ylabel('mean fraction correct (of 32)')
ax.set_ylim(-0.05, 1.1); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
ax.axhline(0.0, color='gray', linestyle=':', linewidth=0.8)

# --- Panel D: example question trajectories side-by-side ---
ax = axes[1][2]
exp_base_ta = EXPS['ternary_adaptive']
exp_base_bi = EXPS['binary']
d0_ta = load_eval(exp_base_ta, 0, 'train_richqa_answer_origqa_frac0.2')
rows0_ta = {r['id']: rs(r) for r in d0_ta['rows']}
all_correct_ta = [sid for sid, r in rows0_ta.items() if r['correct'] == r['total']]

# Find a question that degrades in ternary but stays in binary: id=3307
example_sid = '3307'
for exp_name, exp_base, color, ls in [
    ('binary', exp_base_bi, '#1565C0', '-'),
    ('ternary_adaptive', exp_base_ta, '#E53935', '--'),
]:
    xs, frac_c, frac_n = [], [], []
    for step in STEPS:
        dS = load_eval(exp_base, step, 'train_richqa_answer_origqa_frac0.2')
        if dS is None: continue
        rowsS = {r['id']: rs(r) for r in dS['rows']}
        r = rowsS.get(example_sid)
        if r is None: continue
        xs.append(step)
        frac_c.append(r['correct'] / r['total'])
        frac_n.append(r['not_attempted'] / r['total'])
    ax.plot(xs, frac_c, color=color, linestyle=ls, marker='o', linewidth=2,
            label=f'{exp_name} correct')
    ax.plot(xs, frac_n, color=color, linestyle=':', marker='s', linewidth=1.5, alpha=0.6,
            label=f'{exp_name} IDK')

q_text = next(r['question'] for r in d0_ta['rows'] if r['id'] == example_sid)
ax.set_title(f'Example: id={example_sid}\n"{q_text[:60]}…"', fontsize=8)
ax.set_xlabel('global step'); ax.set_ylabel('fraction of 32 responses')
ax.set_ylim(-0.05, 1.1); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
path = f'{OUT_DIR}/fig9_summary.png'
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'saved → {path}')
