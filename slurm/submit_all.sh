#!/bin/bash
# Submit the pipeline as a dependency chain.
#   bash slurm/submit_all.sh        # from step 0 (re-reads the works TSV)
#   bash slurm/submit_all.sh 1      # papers.tsv already exists: start at step 1
#   bash slurm/submit_all.sh 2      # event tables exist: only panels + analysis
#   SKIP_B=1 bash slurm/submit_all.sh 1   # Option A only
#   REDO_EMB=1 bash slurm/submit_all.sh 1 # force step 0b (re-collect embeddings)
set -euo pipefail
BASE=/N/slate/gpanayio/scisci-temporal
cd "$BASE"
module load python/3.12.4 2>/dev/null || true
mkdir -p logs results
FROM=${1:-0}

# embeddings for Option B: collected once by step 0b (re-run with REDO_EMB=1)
EMB=$(python -c "import config as C; print(C.EMB_NPY)" 2>/dev/null || echo "")

dep() { [ -n "$1" ] && echo "--dependency=afterok:$1" || true; }

J0=""; J1=""
if [ "$FROM" -le 0 ]; then
  J0=$(sbatch --parsable slurm/00-extract-papers.sh); echo "step 0: $J0"
fi
if [ "$FROM" -le 1 ]; then
  J1=$(sbatch --parsable $(dep "$J0") slurm/01-build-events.sh); echo "step 1: $J1"
fi
J0B=""
if [ -z "${SKIP_B:-}" ] && { [ -n "${REDO_EMB:-}" ] || [ ! -f "$EMB" ] || [ "$FROM" -le 0 ]; }; then
  J0B=$(sbatch --parsable $(dep "$J0") slurm/00b-collect-embeddings.sh); echo "step 0b: $J0B"
fi
JA=$(sbatch --parsable $(dep "$J1") slurm/02-option-a.sh); echo "option A: $JA"
JAA=$(sbatch --parsable --dependency=afterok:$JA slurm/04-analyze.sh A); echo "analysis A: $JAA"
if [ -z "${SKIP_B:-}" ]; then
  DEPS=$(echo "$J1 $J0B" | xargs | tr ' ' ':')
  JB=$(sbatch --parsable $(dep "$DEPS") slurm/03-option-b.sh); echo "option B: $JB"
  JBA=$(sbatch --parsable --dependency=afterok:$JB slurm/04-analyze.sh B); echo "analysis B: $JBA"
fi
squeue -u "$USER"
