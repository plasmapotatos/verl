#!/bin/bash
# Quick sanity check: does verl's should_save_ckpt_esi read the env var correctly?
# Run directly (no SLURM needed):
#   bash experiments/rl/test_esi.sh

set -e
cd /work/hdd/bbsg/twei2/rl/verl

apptainer exec --nv --bind /work,/u /work/hdd/bbsg/twei2/rl/torch2501.sif \
  python3 -c "
import os, time, sys

ts_60 = str(int(time.time()) + 60)

# A: deadline 60s away, step=30s, buffer=300s → threshold 390s > 60s remaining → True
os.environ['MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP'] = ts_60
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
a = should_save_ckpt_esi(max_steps_duration=30, redundant_time=300)
print(f'A  deadline=60s  step=30s  buffer=300s  → {a}  (expect True)')

# B: deadline 60s away, step=30s, buffer=0 → threshold 90s > 60s → True
b = should_save_ckpt_esi(max_steps_duration=30, redundant_time=0)
print(f'B  deadline=60s  step=30s  buffer=0s    → {b}  (expect True)')

# C: deadline far away (1h) → False
os.environ['MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP'] = str(int(time.time()) + 3600)
c = should_save_ckpt_esi(max_steps_duration=30, redundant_time=300)
print(f'C  deadline=3600s step=30s buffer=300s  → {c}  (expect False)')

# D: no env var → False
del os.environ['MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP']
d = should_save_ckpt_esi(max_steps_duration=30, redundant_time=300)
print(f'D  no env var                           → {d}  (expect False)')

passed = (a is True and b is True and c is False and d is False)
print()
print('PASS' if passed else 'FAIL')
sys.exit(0 if passed else 1)
"
