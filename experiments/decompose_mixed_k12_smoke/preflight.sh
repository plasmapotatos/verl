#!/usr/bin/env bash
# Preflight checks before running:
#   bash /work/hdd/bbsg/twei2/rl/verl/experiments/decompose_mixed_k12_smoke/run_chain.sh
# Run this on a Delta login node as the user who will launch the chain.

set -u

SIF=/work/hdd/bbsg/twei2/rl/torch2501.sif
W=/work/hdd/bbsg/twei2/rl/verl

fail=0
check() { if eval "$2"; then echo "OK   $1"; else echo "FAIL $1"; fail=1; fi; }

# 1. group + slurm account
check "in delta_bbsg group"        'id | grep -q delta_bbsg'
check "has bbsg-dtai-gh account"   'sacctmgr -n show assoc user=$USER format=Account | grep -q bbsg-dtai-gh'

# 2. container readable + runs
check "sif readable"               'test -r "$SIF"'
echo "---- torch / cuda inside container ----"
apptainer exec --nv --bind /work,/u "$SIF" python -c \
  "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || { echo "FAIL container python"; fail=1; }

# 3. venv activates inside container, verl imports
echo "---- verl import inside container ----"
apptainer exec --bind /work,/u "$SIF" bash -c \
  "source $W/verl_container_rc.sh && python -c 'import verl, transformers, vllm; print(verl.__file__)'" \
  || { echo "FAIL verl import"; fail=1; }

# 4. writable paths
for d in outputs logs chain_state data; do
    check "$W/$d writable" "test -w '$W/$d'"
done

# 5. data present
check "sft train.parquet present" "test -f $W/data/simpleqa/partition/decompose_mixed_k12_smoke/sft/train.parquet"
check "rl  train.parquet present" "test -f $W/data/simpleqa/partition/decompose_mixed_k12_smoke/rl/train.parquet"

# 6. slurm dry-run with the actual sbatch args used by the chain
echo "---- sbatch --test-only ----"
sbatch --test-only \
    --account=bbsg-dtai-gh --partition=ghx4-interactive \
    --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 \
    --gpus-per-node=4 --time=02:00:00 \
    --wrap='echo hi' \
    || { echo "FAIL sbatch dry-run"; fail=1; }

echo
if [[ $fail -eq 0 ]]; then
    echo "ALL CHECKS PASSED — safe to run run_chain.sh"
else
    echo "SOME CHECKS FAILED — fix the FAIL lines above before launching"
    exit 1
fi
