---
mode: agent
description: Create a new plot script under figures/scripts/ using figures/plot_utils. Data is hardcoded in the script. Writes the script, runs it, reports paths.
---

# /make-plot

Create a standalone plot script under `figures/scripts/` that imports and uses `figures/plot_utils` for all styling — never raw `matplotlib` defaults. All data values are hardcoded directly in the script (no runtime file loading).

## Inputs

- `script_name`: filename without extension (e.g. `dilution_curves`). Script written to `figures/scripts/<script_name>.py`.
- `description`: free text describing what the plot should show. If the user has shared data files, read them and extract values to hardcode.

## Hard rules — never break these

1. **Always import from `figures.plot_utils`** — use `make_fig`, `plot_curves`, `style_ax`, `save_fig`. Never call `plt.subplots`, `plt.savefig`, `fig.savefig`, `plt.tight_layout`, or set `plt.rcParams` in the plot script.
2. **Data is always hardcoded.** No `pd.read_parquet`, `open()`, `json.load`, or any runtime file I/O in the script. Read source files yourself first; embed extracted numbers as Python literals.
3. **Output path:** `Path(__file__).parent.parent / "<script_name>.png"`. `save_fig` writes `.png` + `.pdf` and auto-mirrors the PDF to the Overleaf checkout when run with the bind mount.
4. **No `plt.show()`.** Always `save_fig`.
5. **No inline style overrides** (`linewidth=`, `color=`, `fontsize=`) unless the user explicitly requests a deviation.

## Background convention

- Use **white backgrounds** for all figures.
- Achieve this by calling `apply_paper_style(background="white")` (or `apply_bar_rcparams()` for bar plots) from `figures.plot_utils`.
- Do **not** set `plt.rcParams` in the script.

## Naming conventions

- Richqa SFT baseline appears in legends as exactly `RichQA SFT (Baseline)` — no numeric value.

## plot_utils API (do not re-implement)

```python
from figures.plot_utils import apply_paper_style, make_fig, plot_curves, style_ax, save_fig

fig, ax = make_fig()
plot_curves(ax, xs, curves)             # curves = [(label, ys, marker), ...]
style_ax(ax, xlabel, ylabel,
         xticks=None, ylim=None,
         legend_loc="lower right")
save_fig(fig, out_path, dpi=150)        # tight_layout internally
```

## Steps

### 1. Clarify data if needed

If unspecified, ask for: x-values (and meaning), each series' name + y-values + preferred marker (`"o"`, `"s"`, `"^"`, `"D"`), axis labels, y-limits. If the user points to a file, read it yourself and extract values.

### 2. Write the script

```python
"""<one-line description of what this plot shows>

Sources:
    - <series A>: <where the numbers came from>
    - <series B>: <where the numbers came from>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from figures.plot_utils import make_fig, plot_curves, style_ax, save_fig

XS = [...]
CURVES = [
    ("Series A", [...], "o"),
    ("Series B", [...], "s"),
]
OUT = Path(__file__).parent.parent / "<script_name>.png"


def main():
  apply_paper_style(background="white")
    fig, ax = make_fig()
    plot_curves(ax, XS, CURVES)
    style_ax(ax, xlabel="...", ylabel="...", xticks=XS, ylim=(0, None))
    save_fig(fig, OUT)


if __name__ == "__main__":
    main()
```

- Include a `Sources:` block noting where each series' numbers came from.
- `ylim` lower bound = 0 unless data warrants otherwise.
- `xticks=XS` when x-values are a small discrete list.

### 3. Run the script

```bash
cd /work/hdd/bbsg/twei2/rl/verl && \
apptainer exec --bind /work/hdd/bbsg/twei2/rl/69e95abe7a34d3bbc17ab21e \
  /work/hdd/bbsg/twei2/rl/torch2501.sif python figures/scripts/<script_name>.py
```

The `--bind` is required so `save_fig` can mirror the PDF into the Overleaf checkout (which lives outside the default container mount).

If it errors, fix and re-run. Do not report success until the image is actually saved.

### 4. Report

- Script path: `figures/scripts/<script_name>.py`
- Image paths: `figures/<script_name>.{png,pdf}`
- Overleaf copy: `assets/figures/<script_name>.pdf`
- Regenerate one-liner (must include `--bind`)
- Paste-ready commit+push for when the user is done iterating (do NOT run yourself):
  ```
  cd /work/hdd/bbsg/twei2/rl/69e95abe7a34d3bbc17ab21e && \
    git add assets/figures/<script_name>.pdf && \
    git commit -m "update <script_name> figure" && \
    git push
  ```
