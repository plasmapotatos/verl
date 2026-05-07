# GitHub Copilot — repo instructions

GitHub Copilot Chat auto-loads this file as context for every chat in this repository.

## Environment

All Python commands must run inside the Apptainer container:

```bash
apptainer exec /work/hdd/bbsg/twei2/rl/torch2501.sif <command>
```

For plot scripts that need to mirror PDFs to the Overleaf checkout, add the bind mount:

```bash
apptainer exec --bind /work/hdd/bbsg/twei2/rl/69e95abe7a34d3bbc17ab21e \
  /work/hdd/bbsg/twei2/rl/torch2501.sif python figures/scripts/<name>.py
```

Container Python: `/usr/bin/python` (3.12.3).

## Codebase

Before suggesting a solution, check whether the feature already exists (grep/find). Don't reinvent.

## Experiment analyses

Write experiment analyses as `ANALYSIS_*.md` inside the experiment's checkpoint/output directory. Always include concrete failure-mode and success-mode samples (actual inputs, outputs, scores) — not just aggregate metrics.

## Making figures for the paper

When the user asks for a new figure:

- Create a standalone script in `figures/scripts/<name>.py` that imports `figures.plot_utils` (`make_fig`, `plot_curves`, `style_ax`, `save_fig`) for all styling. Never call raw matplotlib (`plt.subplots`, `plt.savefig`, `plt.tight_layout`, `plt.rcParams`) in the plot script.
- All data values are **hardcoded** in the script. No runtime `pd.read_parquet`/`open()`/`json.load`. Read source files yourself first, extract numbers, embed as Python literals.
- Image output: `Path(__file__).parent.parent / "<name>.png"`. `save_fig` writes both `.png` and `.pdf`, and auto-mirrors the PDF into the Overleaf checkout when run with the bind mount.
- No `plt.show()`. No inline style kwargs (`linewidth=`, `color=`, `fontsize=`) unless the user explicitly asks for a deviation from defaults.
- Legend convention: refer to the richqa SFT baseline as exactly `RichQA SFT (Baseline)` — no numeric value.

For the full template and step-by-step workflow, see `.github/prompts/make-plot.prompt.md` (invoke as `/make-plot` in Copilot Chat).
