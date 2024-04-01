#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=1907
#SBATCH --array=1-174
#SBATCH --time=300
#SBATCH --output=/home/schmidtfa/git/pyrasa/paper/jobs/005/log/out_%a.log

echo "Executing Job $SLURM_JOB_ID on $SLURMD_NODENAME"
/home/schmidtfa/miniconda3/envs/pyrasa/bin/python /home/schmidtfa/git/pyrasa/paper/jobs/005/slurm/runner.py