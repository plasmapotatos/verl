#!/usr/bin/env bash
# Minimal launcher for the bracket-substitution SFT trainer.
#
# Required env:
#   TRAIN_DATA         parquet path
#   VAL_DATA           parquet path
#   ANCHOR_BANK        txt file, one anchor per line (see scripts/extract_factual_anchors.py)
#
# Optional:
#   BASE_MODEL         default Qwen/Qwen2.5-3B-Instruct
#   PROJECT_NAME       default bracket_substitution_sft
#   EXPERIMENT_NAME    default test
#   MAX_LENGTH         default 4096
#   EPOCHS             default 3
#   LR                 default 1.5e-4
#   NPROC              default 1
#
# Backward compat: the base sft_trainer.yaml is untouched. All mode-specific
# fields are injected via Hydra `+` overrides.
set -euo pipefail

VERL_DIR="${VERL_DIR:-/work/hdd/bbsg/twei2/rl/verl}"
APPTAINER_IMG="${APPTAINER_IMG:-/work/hdd/bbsg/twei2/rl/torch2501.sif}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PROJECT_NAME="${PROJECT_NAME:-bracket_substitution_sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-test}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1.5e-4}"
NPROC="${NPROC:-1}"

: "${TRAIN_DATA:?TRAIN_DATA required}"
: "${VAL_DATA:?VAL_DATA required}"
: "${ANCHOR_BANK:?ANCHOR_BANK required}"

cd "$VERL_DIR"

apptainer exec "$APPTAINER_IMG" \
    torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    -m verl.trainer.bracket_sft_trainer \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.prompt_key=question \
    data.response_key=answer \
    data.max_length="$MAX_LENGTH" \
    data.truncation=right \
    data.filter_overlong_prompts=True \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=2 \
    +data.custom_cls.path=verl/utils/dataset/bracket_sft_dataset.py \
    +data.custom_cls.name=BracketSFTDataset \
    +data.bracket_substitution.enable=true \
    +data.bracket_substitution.anchor_bank_path="$ANCHOR_BANK" \
    +data.bracket_substitution.substitute_in_eval=false \
    model.partial_pretrain="$BASE_MODEL" \
    optim.lr="$LR" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs="$EPOCHS" \
    trainer.logger='[console,wandb]' \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=False
