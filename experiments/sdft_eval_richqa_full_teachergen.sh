#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_DIR="${OUTPUT_DIR:-/work/hdd/bbsg/twei2/rl/Self-Distillation/outputs/sdft_richqa_full_3b_teachergen}"
exec bash "$(dirname "$0")/sdft_eval_richqa_full.sh"
