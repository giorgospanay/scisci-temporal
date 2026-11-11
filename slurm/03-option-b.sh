#!/bin/bash

#SBATCH -J option-b
#SBATCH -o logs/option-b_%j.txt
#SBATCH -e logs/option-b_%j.err
#SBATCH --chdir=/N/slate/gpanayio/scisci-temporal
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH -A r02320

module load python/3.12.4
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "===== [2B] Role activation panel (B_collab + B_sim): started $(date) ====="
python -u 03_option_b_role_activation.py
echo "===== [2B] finished $(date) ====="
