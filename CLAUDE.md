# Claude Code Instructions

## Environment

All Python commands must be run inside the Apptainer container. Always include
`--bind /work/hdd/bbsg/twei2/rl/69e95abe7a34d3bbc17ab21e` so the Overleaf paper
checkout is visible from inside the container (needed for `save_fig` to mirror
PDFs into `assets/figures/`):

```
apptainer exec --bind /work/hdd/bbsg/twei2/rl/69e95abe7a34d3bbc17ab21e /work/hdd/bbsg/twei2/rl/torch2501.sif <command>
```

Examples:
- `apptainer exec --bind /work/hdd/bbsg/twei2/rl/69e95abe7a34d3bbc17ab21e /work/hdd/bbsg/twei2/rl/torch2501.sif python script.py`
- `apptainer exec --bind /work/hdd/bbsg/twei2/rl/69e95abe7a34d3bbc17ab21e /work/hdd/bbsg/twei2/rl/torch2501.sif pip list`

Container Python: `/usr/bin/python` (Python 3.12.3)

- Python is the primary language. Use simple, readable solutions — avoid over-engineering with complex statistical methods or tiered systems unless explicitly requested.

## Data Analysis section in CLAUDE.md.\n\n## Data Analysis Conventions
- Before loading data files, confirm which files correspond to train vs val splits. Ask if unsure rather than guessing.
- When referencing experiment results, verify the data source path with the user before producing analysis.

## Interaction Guidelines section in CLAUDE.md.\n\n## Codebase Awareness
- Before suggesting solutions, check if the feature already exists in the codebase. Use Grep/Read to verify.

## Experiment Analysis
- When analyzing an experiment, save the written analysis as a markdown file inside the experiment's checkpoint/output
    directory.
- Always include concrete samples of failure modes and success modes (actual inputs, outputs, scores) — not just
aggregate metrics — so the reasoning is auditable later.
