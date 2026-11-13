#!/bin/bash

#SBATCH -J collect-embeddings
#SBATCH -o logs/collect-embeddings_%j.txt
#SBATCH -e logs/collect-embeddings_%j.err
#SBATCH --chdir=/N/slate/gpanayio/scisci-temporal
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH -A r02320

module load python/3.12.4
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "===== [0b] Collect SPECTER2 chunks into one array: started $(date) ====="
python -u 00b_collect_embeddings.py
echo "===== [0b] finished $(date) ====="
