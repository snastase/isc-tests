#!/bin/bash

# Run from isc-tests directory:
# sbatch fpr_grid_slurm.sh

# Set partition
#SBATCH --partition=all

# How long is job (in minutes)?
#SBATCH --time=720

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=4 --mem-per-cpu=12000

# Name of jobs?
#SBATCH --job-name=fpr_grid

# Where to output log files?
#SBATCH --output='logs/fpr_grid_loo_%A_%a.log'

# Number jobs to run in parallel, pass index as seed
#SBATCH --array=1-1000

# Remove modules just to make sure
echo "Purging modules"
module purge

# Load my python3 conda environment
echo "Loading local python3 conda environment"
source /usr/people/snastase/.bashrc 
conda activate python3

# Print job submission info
echo "Slurm job ID: " $SLURM_JOB_ID
echo "Slurm array task ID: " $SLURM_ARRAY_TASK_ID
date

# Run with simulation seed based on array index
echo "Running ISC simulation with seed $SLURM_ARRAY_TASK_ID"

./fpr_grid.py $SLURM_ARRAY_TASK_ID

echo "Finished running ISC simulation with seed $SLURM_ARRAY_TASK_ID"
date
