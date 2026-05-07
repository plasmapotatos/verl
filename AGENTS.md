# Repo agent instructions

This file is read by AGENTS.md-aware agents (Codex CLI, Cursor, etc.) at the start of every session.

## Environment

All Python commands must be run inside the Apptainer container:

```bash
apptainer exec /work/hdd/bbsg/twei2/rl/torch2501.sif <command>
```

For plot scripts (which auto-mirror PDFs into the Overleaf checkout), add a bind mount:

```bash
apptainer exec --bind /work/hdd/bbsg/twei2/rl/69e95abe7a34d3bbc17ab21e \
  /work/hdd/bbsg/twei2/rl/torch2501.sif python figures/scripts/<name>.py
```

Container Python: `/usr/bin/python` (3.12.3).

## Codebase awareness

Before suggesting a solution, check whether the feature already exists. Use grep/find to verify.

## Experiment analyses

When analyzing an experiment, write the analysis as `ANALYSIS_*.md` inside the experiment's checkpoint/output directory. Always include concrete failure-mode and success-mode samples (actual inputs, outputs, scores) — not just aggregate metrics.

---

# Plot-making convention ("make-plot")

When the user asks you to make a new figure for the paper, follow these rules. The behavior matches the `/make-plot` slash command in our Claude Code workflow.

Create a standalone plot script under `figures/scripts/` that imports and uses `figures/plot_utils` for all styling — never raw `matplotlib` defaults. All data values are hardcoded directly in the script (no runtime file loading).

## Hard rules — never break these

1. **Always import from `figures.plot_utils`** — use `make_fig`, `plot_curves`, `style_ax`, `save_fig`. Never call `plt.subplots`, `plt.savefig`, `fig.savefig`, `plt.tight_layout`, or set `plt.rcParams` directly in the script.
2. **Data is always hardcoded** — no `pd.read_parquet`, `open()`, `json.load`, or any runtime file I/O in the plot script. Read source files yourself first, extract the numbers, and embed them as Python literals.
3. **Output path** — images go in `figures/` (one level up from the script): `Path(__file__).parent.parent / "<script_name>.png"`. `save_fig` writes both `.png` and `.pdf` and (when the bind mount is present) auto-mirrors the PDF into the Overleaf checkout.
4. **No `plt.show()`** — scripts run headlessly; always use `save_fig`.
5. **No inline style overrides** — no `linewidth=`, `color=`, `fontsize=` kwargs unless the user explicitly requests a deviation from defaults.

## Naming conventions

- The richqa SFT baseline must appear in legends as exactly `RichQA SFT (Baseline)` — no numeric value in the label.

## plot_utils API (do not re-implement)

```python
from figures.plot_utils import make_fig, plot_curves, style_ax, save_fig

fig, ax = make_fig()                    # default figsize
plot_curves(ax, xs, curves)             # curves = [(label, ys, marker), ...]
style_ax(ax, xlabel, ylabel,
         xticks=None, ylim=None,
         legend_loc="lower right")
save_fig(fig, out_path, dpi=150)        # tight_layout internally; writes png + pdf; mirrors pdf to Overleaf
```

## Workflow

1. **Clarify data if needed.** If the user hasn't specified the data, ask for: x-axis values (and what they represent), each series' name + y-values + preferred marker (`"o"`, `"s"`, `"^"`, `"D"`), axis labels, y-limits. If the user points to a file (parquet/CSV/JSON), read it yourself and extract the values — do not load files at runtime in the script.

2. **Write the script** at `figures/scripts/<script_name>.py`:

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
       fig, ax = make_fig()
       plot_curves(ax, XS, CURVES)
       style_ax(ax, xlabel="...", ylabel="...", xticks=XS, ylim=(0, None))
       save_fig(fig, OUT)

   if __name__ == "__main__":
       main()
   ```

   - `Sources:` docstring block notes where each series' numbers came from.
   - `ylim` lower bound = 0 unless data warrants otherwise.
   - `xticks=XS` when x-values are a small discrete list.

3. **Run the script** (note the `--bind` for Overleaf auto-copy):

   ```bash
   cd /work/hdd/bbsg/twei2/rl/verl && \
   apptainer exec --bind /work/hdd/bbsg/twei2/rl/69e95abe7a34d3bbc17ab21e \
     /work/hdd/bbsg/twei2/rl/torch2501.sif python figures/scripts/<script_name>.py
   ```

   If it errors, fix and re-run. Do not report success until the image is actually saved.

4. **Report**:
   - Script path: `figures/scripts/<script_name>.py`
   - Image paths: `figures/<script_name>.{png,pdf}`
   - Overleaf copy: `assets/figures/<script_name>.pdf`
   - Regenerate one-liner: `python figures/scripts/<script_name>.py` (from repo root inside container, with the `--bind`)
   - Paste-ready commit+push (do NOT run this yourself — let the user trigger it):
     ```
     cd /work/hdd/bbsg/twei2/rl/69e95abe7a34d3bbc17ab21e && \
       git add assets/figures/<script_name>.pdf && \
       git commit -m "update <script_name> figure" && \
       git push
     ```
