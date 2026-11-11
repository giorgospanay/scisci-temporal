#!/bin/bash

#SBATCH -J option-a
#SBATCH -o logs/option-a_%j.txt
#SBATCH -e logs/option-a_%j.err
#SBATCH --chdir=/N/slate/gpanayio/scisci-temporal
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH -A r02320

module load python/3.12.4
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "===== [2A] Field entry panel: started $(date) ====="
python -u 02_option_a_field_entry.py
echo "===== [2A] finished $(date) ====="
