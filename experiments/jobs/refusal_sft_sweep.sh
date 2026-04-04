#!/bin/bash

JOBID_75_25=$(sbatch --parsable experiments/simpleqa/sft/batch_75_25.slurm)
echo "Submitted 75_25 job: $JOBID_75_25"

JOBID_25_75=$(sbatch --parsable experiments/simpleqa/sft/batch_25_75.slurm)
echo "Submitted 25_75 job: $JOBID_25_75"