#!/bin/bash

#SBATCH -J analyze
#SBATCH -o logs/analyze_%j.txt
#SBATCH -e logs/analyze_%j.err
#SBATCH --chdir=/N/slate/gpanayio/scisci-temporal
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -A r02320

module load python/3.12.4
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# usage: sbatch slurm/04-analyze.sh A   (or B)
OPT=${1:-A}
echo "===== [3] Analysis, option $OPT: started $(date) ====="
python -u 04_analyze.py "$OPT"
echo "===== [3] finished $(date) ====="
