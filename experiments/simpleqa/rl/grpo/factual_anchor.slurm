#!/bin/bash
#SBATCH --job-name=simpleqa_factual_anchor_grpo
#SBATCH --account=bbsg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gpus-per-node=2
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/simpleqa_factual_anchor_grpo_%j.out
#SBATCH --error=logs/simpleqa_factual_anchor_grpo_%j.err

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

srun $VERL bash -c 'source /work/hdd/bbsg/twei2/rl/verl_container_rc.sh && \
                    cd /work/hdd/bbsg/twei2/rl/verl && \
                    bash experiments/simpleqa/rl/grpo/factual_anchor.sh'

echo "Job finished at $(date)"
