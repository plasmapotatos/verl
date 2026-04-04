#!/bin/bash
#SBATCH --job-name=sft_75_25
#SBATCH --account=bbsg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gpus-per-node=2
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/sft_75_25_%j.out
#SBATCH --error=logs/sft_75_25_%j.err

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

srun $VERL bash -c 'source /work/hdd/bbsg/twei2/rl/verl_container_rc.sh && \
                    cd /work/hdd/bbsg/twei2/rl/verl && \
                    bash experiments/simpleqa/sft/run_simpleqa_rich_sft_progressive_75_25.sh'

srun $VERL bash -c 'source /work/hdd/bbsg/twei2/rl/verl_container_rc.sh && \
                    cd /work/hdd/bbsg/twei2/rl/verl && \
                    bash experiments/simpleqa/rl/grpo/richqa_refusal_ternary_adaptive_75_25.sh'

echo "Job finished at $(date)"