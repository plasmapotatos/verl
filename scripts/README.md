# Script guide

> ⚠️ Many examples hard-code `/work/hdd/bbsg/twei2/rl/verl` and other absolute paths; point them at your repo (or use relative paths) before you run anything.

## Projects & experiments
A **project** is a collection of experiments under `outputs/sft/<project>` or `outputs/rl/<project>`. Each **experiment** is one training run with its own `global_step_*` checkpoints, pass@k results, and plots. The scripts below assume this layout so they can find generations, rollouts, and pass@k folders consistently.

## Training
### SFT sweeps
- `experiments/sft/run_sft_sweep_with_eval.sh` is the entry point for sweeping epochs and learning rates. It runs training, merges the last checkpoint, generates eval outputs (optional `EVAL_DATA` list), grades them through `verl.eval.cli`, and triggers pass@k and plotting.
  - **Main inputs**: `PROJECT_NAME`, `TRAIN_DATA`, optional `EVAL_DATA` (space-separated parquet paths)
  - **Key knobs**: `LR_LIST`, `EPOCHS_LIST`, `PASS_AT_K_MODE` (controls whether pass@k runs at all/last/all checkpoints).
  - **Outputs**: `outputs/sft/<project_name>/<experiment>/global_step_*` for checkpoints, `generations` subdirs, `pass@k` summaries, `plots` directory when `EVAL_DATA` exists.

- `experiments/sft/run_sft_progressive_eval.sh` keeps training running up to the highest epoch in `EPOCHS_LIST` while polling for checkpoints that exceed each epoch target. It runs eval/gen/grade for each checkpoint it detects and then prunes stale checkpoints automatically.
  - **Main inputs**: `PROJECT_NAME`, `TRAIN_DATA`, optional `EVAL_DATA`, `PASS_AT_K_EVAL_DATA` (space-separated parquet paths).
  - **Key knobs**: `LR_LIST`, `EPOCHS_LIST`
  - **Outputs**: Same layout as the sweep script plus scheduler files `evaluated_epochs.txt`/`evaluated_checkpoints.txt`, and plots under `outputs/sft/<project>/plots`.

### RL
- `experiments/rl/grpo.sh` trains/evaluates GRPO on a project. It feeds `verl.trainer.main_ppo` with rollouts, generates evals with `experiments/utils/eval_all_checkpoints.sh`, and runs pass@k via `scripts/run_pass_at_k_experiment.py`. By default it also scores the base Qwen model and prunes occluded checkpoints.
  - **Main inputs**: `PROJECT_NAME`, `TRAIN_DATA`, `MODEL_PATH` (pretrained SFT checkpoint).
  - **Key knobs**: `TOTAL_EPOCHS`, `ROLLOUT_N`, `PASS_AT_K_TOP_K` (number of samples per prompt in pass@k).
  - **Outputs**: `outputs/rl/<project_name>/<experiment_name>/global_step_*`, `pass@k/<dataset>/`, `rollouts/`, `plots/`, and base resolutions under `outputs/rl/.../base`.
  - `experiments/simpleqa/rl/grpo/rich_sft_train_eval.sh` is a curated wrapper that fills these env vars for the SimpleQA rich SFT data before invoking `grpo.sh`.

- `experiments/rl/spin_dpo.sh` is the SPIN-DPO counterpart that runs `recipe.spin.main_spin`, evals with the same utility, and prunes as needed.
  - **Main inputs**: `PROJECT_NAME`, `TRAIN_DATA`, `MODEL_PATH` (SFT checkpoint used as actor reference).
  - **Key knobs**: `TOTAL_EPOCHS`, `ROLLOUT_N`, `PPO_MINI_BATCH_SIZE`.
  - **Outputs**: `outputs/rl/<project>/<exp>/global_step_*`, eval generations, and `plots`.
  - `experiments/simpleqa/rl/spin_dpo/run_simpleqa_rich_sft_rl.sh` shows how to wire the SimpleQA rich SFT data into this flow.

## Data processing
- **Parquet-first workflow**: All scripts consume and emit parquet files (download a parquet viewer for debugging). The base SimpleQA dataset lives at `data/simpleqa/data.parquet` and a full copy is available under `/tmp/data` if you need to stage experiments elsewhere.
- **Typical flow**:
  1. Restore each augmented dataset to the original QA text via `scripts/restore_simpleqa_original_qa.py`.
  2. Sample eval/train splits with `scripts/make_eval_subset.py` (keep the remainder by passing `--save-remainder`).
  3. Combine shards with `scripts/combine_parquet.py` when you need a single file.
  4. Feed the resulting train/eval parquet file into the SFT/RL training endpoints and use `make_eval_subset.py` again when you want a smaller evaluation slice (e.g., to judge on a tiny subset).
- **Scripts**:
  - `scripts/make_eval_subset.py` – `{--input,<path>}, {--fraction,0.1}, {--output,<path>}`, optional `--save-remainder/--train-output`. Outputs a sampled parquet plus optional remainder.
  - `scripts/combine_parquet.py` – `--input_paths` (one or more parquets), `--output_dir`, `--seed` and `--no-shuffle` to keep order. Outputs `<target_dir>/combined.parquet`.
  - `scripts/restore_simpleqa_original_qa.py` – `--input` (augmented parquet), `--base-parquet` (defaults to `data/simpleqa/data.parquet`), optional `--output`. Emits a copy where each row inherits question/answer/prompt from the canonical dataset.
- These scripts avoid in-place edits so you can chain them without overwriting the raw data.

## Augmentation
Augmentation scripts live under `verl.augment.cli` (see its own README). The TL;DR command is:

```bash
python -m verl.augment.cli \
  --input /path/to/input.parquet \
  --output /path/to/output.parquet \
  --method templates
```

Attach the desired method and tune the CLI arguments documented in `verl/augment/cli.py` for more control.

## Eval
- **Base eval**: Use `verl.eval.cli` for single-shot grading (the training scripts call it internally). Most experiments only need to pass `--dataset simpleqa`, `--input <gen>.parquet`, and `--output <eval>.json`.
- `scripts/pass_at_k.py` – required args `--checkpoint`, `--eval-data`, `--output-dir`, `--top-k`; optional `--top-p`, `--temperature`, `--n-gpus`, `--limit`, `--use-judge`. Generates `pass@k/generations_<k>.parquet`, `eval_<k>.json`, and `pass_at_k_<k>.json` summaries.
- `scripts/run_pass_at_k_experiment.py` – loops over every merged checkpoint under `--experiment-dir`, feeding them into `pass_at_k.py`. Main inputs: `--experiment-dir`, `--eval-data`, `--ks` (comma-separated), optional `--output-dir` / `--layout`. Outputs per-checkpoint `pass@k/<dataset>/pass_at_k.json` plus a consistent folder structure for multiple k values.

## Postprocess
- `scripts/plot_sft_eval_metrics.py` – feeds `outputs/<project>` or a single experiment directory to generate PNGs per dataset (`<dataset>/<metric>_by_step.png`) plus a combined accuracy/pass@k plot under `combined/`. Requires `--project-dir` and `--output-dir`.
- `scripts/analyze_grpo_rollouts.py` – takes `--rollout-dir` (JSONL rollouts) and `--dataset` (parquet) and outputs two files: a JSON bundle (default `outputs/rollout_analysis/<rollout_name>.json`) and an HTML explorer (default `...html`). Additional options: `--sample-id`, `--sample-index`, `--question`, `--max-rollouts-per-step`. The HTML lets you browse rewards/outputs per step.

## Other
- `scripts/prune_project_checkpoints.sh PROJECT_DIR [--dry-run] [--all]` – iterates over every experiment that contains `global_step_*`, calls `scripts/prune_experiment_checkpoints.py`, and keeps only the last `.pt` checkpoint per experiment (all `.safetensors` files stay). Beware: no old checkpoints remain for resuming training.

## Notes
- Always run the wrappers from the repo root so the relative `outputs/` and `data/` paths resolve.
- Projects consume the standard SimpleQA dataset (`data/simpleqa/data.parquet`); augmentations should feed into `restore_simpleqa_original_qa.py` before training so the QA text and reward models line up with evaluation rules.
- Before training, use `make_eval_subset.py` to snapshot a small eval set and optionally save the remainder for quick train-fit checks with the judge.
