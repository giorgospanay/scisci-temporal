# scisci-temporal: becoming a bridge (interaction / activation windows)

## Idea

We adapt the design of Venturini et al. (*Sci. Rep.* 14:1258, 2024) from topic switches to
interdisciplinarity. For every reference year T0 (1995–2018, one per year), the **interaction
window** IW = [T0−5, T0) is where we observe who an author collaborates with. The **activation
window** AW = [T0, T0+5) is where we check whether they "turn active". Windows slide by one year,
so consecutive T0s overlap. Results are aggregated over T0 with bootstrap CIs, and the regression
clusters by author.

We define "active" in two ways. Both are built around CS authors.

- **Option A, field entry.** A CS author who has never published in discipline X activates
  when their first X paper falls in the AW. X ∈ {Math, Physics, Biology, Sociology, Economics,
  Linguistics}. Exposure = joint IW papers with co-authors already active in X.
- **Option B, role activation.** B_collab and B_sim are recomputed per window. A CS author
  who is a coordinator in the IW activates if they are a bridge in the AW, meaning high B_collab
  **or** high B_sim (gatekeeper, representative or liaison). "High" means above the 80th
  percentile of authors with the same home discipline in the same window. "Coordinator" means at
  or below the 60th percentile on both layers. Authors in between are dropped, so threshold noise
  is not counted as activation. Exposure = joint IW papers with co-authors who are bridges in the
  IW.

## Data

| What | Where |
|---|---|
| SPECTER2 chunks (input) | `/N/scratch/gpanayio/embeddings/` |
| Works TSV (input) | `/N/scratch/gpanayio/openalex-pre/works_core+...+mesh.tsv` |
| Large intermediates | `/N/scratch/gpanayio/scisci-temporal/` (scratch is purged after inactivity) |
| Code, logs, results | `/N/slate/gpanayio/scisci-temporal/` |

Papers are articles from 1990–2022 with at most 30 authors. Each paper is tagged with every one
of the seven level-0 concepts it has at score ≥ 0.3, as in `filter-papers.py`. A paper with none
of the seven is dropped. An author's **home** is the discipline with the largest fractional
paper count. The population is authors with ≥ 10 papers and ≥ 5 distinct co-authors. Networks
are reconstructed for CS-home authors plus their population co-authors (`FOCUS_HOMES`,
`EXPAND_NEIGHBOURS`).

## Pipeline

| Step | Script | Slurm | Resources | What it does |
|---|---|---|---|---|
| 0 | `00_extract_papers.py` | `00-extract-papers.sh` | 1 CPU, 32 GB | Two passes over the works TSV. Writes `papers.tsv` (id, year, authors, disciplines) and citation counts for every kept paper |
| 0b | `00b_collect_embeddings.py` | `00b-collect-embeddings.sh` | 64 GB | Reads the SPECTER2 chunks (`embeddings_<k>.npy` + `ids_<k>.txt`) and writes one float16 array holding only the papers from step 0, without duplicates. Needed for Option B only |
| 1 | `01_build_events.py` | `01-build-events.sh` | 256 GB | Home disciplines, population, co-authorship **event list** (one row per joint paper, so any window is a year filter), per-author field activity by year |
| 2A | `02_option_a_field_entry.py` | `02-option-a.sh` | 128 GB | Candidate × T0 panel for each CS→X, plus the contact and source tables |
| 2B | `03_option_b_role_activation.py` | `03-option-b.sh` | 16 CPUs, 256 GB | Windowed B_collab and B_sim, coordinator/bridge states, same tables |
| 3 | `04_analyze.py A` / `B` | `04-analyze.sh A` / `B` | 64 GB | Results and figures in `results/results_A`, `results/results_B` |

### Running

```bash
cd /N/slate/gpanayio/scisci-temporal
bash slurm/setup_env.sh               # once: pyarrow, statsmodels, faiss-cpu, ...
# edit config.py: MIN_CONCEPT_SCORE if filter-papers.py used another value
bash slurm/submit_all.sh              # whole chain with dependencies
bash slurm/submit_all.sh 1            # skip step 0 if papers.tsv already exists
SKIP_B=1 bash slurm/submit_all.sh     # Option A only (no embeddings needed)
```

Step 0b runs after step 0 and in parallel with step 1. It is skipped when
`embeddings_needed.npy` already exists, unless you set `REDO_EMB=1`. Steps 2A and 2B run in
parallel after step 1, and 2B also waits for 0b.

### B_sim per window

Each author's window centroid (the mean embedding of their papers in that window) is queried
against **one fixed index** of all paper embeddings, so nothing is re-embedded and no
similarity graph is rebuilt. Of the nearest papers, the author's own papers and papers
published after the window are dropped. The top 20 that remain give a Blau index over their
disciplines. Step 0b builds the index input from the chunk folder. `EMBEDDING_FILES` can instead point to
other `(embeddings.npy, ids)` pairs.

- Up to 5M vectors: exact faiss index.
- Above that: compressed IVF-PQ index (about 64 bytes per paper).

On test data the compressed index reproduced exact B_sim with correlation 0.998. This
approximates the static B_sim, so compare the two on the final window before relying on it.

## Analyses (per focal set; bootstrap CIs across T0)

1. **Target activation** C(k): the share of candidates with ≥ k exposures who activate,
   compared with the independent-contacts baseline 1 − (1 − p)^k.
2. **Prominence, target side**: C(k) for candidates exposed exclusively to top-10% sources
   minus those exposed exclusively to bottom-10% sources, by productivity and by impact.
3. **Source side**: activation rate of each source's exclusive inactive co-authors, top minus
   bottom 10%. Option A also reports chaperoning: the first X paper is written with that source.
4. **Dilution**: among the top-10% sources, the source-activation difference between sources
   whose exclusive co-authorships have large vs small teams.
5. **Logit** with author-clustered SEs and T0 fixed effects. Exposure × degree, career age,
   productivity, team size. In Option B, the counts of liaison, gatekeeper and representative
   neighbours.
6. **Stratified null**: activation is permuted within (T0, home, degree quintile, productivity
   quintile).

## Deviations from Venturini et al. (state them in the methods)

- **Candidates must publish in the AW.** "Did not activate" is not the same as "stopped
  publishing".
- **Unexposed candidates (k = 0) are kept.** They give the spontaneous-activation rate.
- **Impact** is the mean all-time citations of the source's IW papers in the focal field, taken
  from the TSV's reference lists.
- **Option B thresholds are relative to peers in the same window and discipline.** This nets out
  the secular growth of teams and cross-field ties.

## Caveats

- **Non-CS co-authors are a selected subset.** With `FOCUS_HOMES = ["CS"]`, the only non-CS
  authors reconstructed are those who co-author with CS. Their percentile cuts are therefore
  relative to that subset, which is likely more interdisciplinary than their discipline as a
  whole. Set `FOCUS_HOMES = None` to reconstruct everyone. This also enables X→CS in Option A,
  at a higher cost.
- **Selection and influence are confounded** (Shalizi & Thomas 2011). The null model and
  controls narrow this gap but do not close it. The next step is a reverse-time placebo.
- **Concept tags and author disambiguation are noisy.** Very high-degree authors may be merged
  identities.

## Smoke test

`python make_toy_data.py /tmp/toy` writes a synthetic works TSV in the same column layout, plus
two embedding files. Point the pipeline at it with environment variables:

```bash
export WORKS_FILE=/tmp/toy/works.tsv OUT_DIR=/tmp/toy/out RESULTS_DIR=/tmp/toy/results \
       MIN_FIELD_PAPERS_PER_WINDOW=300 \
       EMBEDDING_FILES="/tmp/toy/emb_0.npy,/tmp/toy/emb_0_ids.txt;/tmp/toy/emb_1.npy,/tmp/toy/emb_1_ids.txt"
# (or set EMBEDDING_CHUNK_DIR to a folder of embeddings_<k>.npy / ids_<k>.txt and run 00b_collect_embeddings.py)
python 00_extract_papers.py && python 01_build_events.py
python 02_option_a_field_entry.py && python 04_analyze.py A
python 03_option_b_role_activation.py && python 04_analyze.py B
```
