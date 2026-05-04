#!/usr/bin/env bash
set -euo pipefail

bash /work/hdd/bbsg/twei2/rl/verl/experiments/sdft_eval_and_plot.sh \
    /work/hdd/bbsg/twei2/rl/verl/outputs/sdft_simpleqa_window_3b_teachergen \
    /work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/decompose/sft/train_eval.parquet
