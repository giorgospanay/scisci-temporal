#!/bin/bash
# One-off: install the Python packages the pipeline needs (login node is fine).
module load python/3.12.4
python -m pip install --user pandas numpy pyarrow statsmodels matplotlib faiss-cpu
python -c "import pandas, pyarrow, statsmodels, matplotlib, faiss; print('environment OK')"
