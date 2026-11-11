#!/bin/bash

#SBATCH -J extract-papers
#SBATCH -o logs/extract-papers_%j.txt
#SBATCH -e logs/extract-papers_%j.err
#SBATCH --chdir=/N/slate/gpanayio/scisci-temporal
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH -A r02320

module load python/3.12.4
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "===== [0] Extract papers from works TSV: started $(date) ====="
python -u 00_extract_papers.py
echo "===== [0] finished $(date) ====="
