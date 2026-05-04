#!/usr/bin/env bash
set -euo pipefail
cd /work/hdd/bbsg/twei2/rl/verl
python /work/hdd/bbsg/twei2/rl/verl/scripts/compare_refusal_on_unknown.py "$@"
