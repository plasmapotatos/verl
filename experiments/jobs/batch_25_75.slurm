#!/bin/bash
#SBATCH --job-name=sft_25_75
#SBATCH --account=bbsg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gpus-per-node=2
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/sft_25_75_%j.out
#SBATCH --error=logs/sft_25_75_%j.err

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

srun $VERL bash -c 'source /work/hdd/bbsg/twei2/rl/verl_container_rc.sh && \
                    cd /work/hdd/bbsg/twei2/rl/verl && \
                    bash experiments/simpleqa/sft/run_simpleqa_rich_sft_progressive_25_75.sh'

srun $VERL bash -c 'source /work/hdd/bbsg/twei2/rl/verl_container_rc.sh && \
                    cd /work/hdd/bbsg/twei2/rl/verl && \
                    bash experiments/simpleqa/rl/grpo/richqa_refusal_ternary_adaptive_25_75.sh'

echo "Job finished at $(date)"