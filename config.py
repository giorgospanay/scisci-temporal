"""
Configuration for the activation-window pipeline (adapted from
Venturini et al., Sci. Rep. 14:1258, 2024).

Paths default to the cluster layout. Every path can be overridden with an
environment variable of the same name (the toy test does this), e.g.
    WORKS_FILE=toy_works.tsv OUT_DIR=toy_out python 00_extract_papers.py
"""
import os
from pathlib import Path


def _env(name, default):
    return os.environ.get(name, default)


# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE = Path(_env("BASE", "/N/slate/gpanayio/scisci-temporal"))
WORKS_FILE = _env("WORKS_FILE",
                  "/N/scratch/gpanayio/openalex-pre/"
                  "works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv")
# large intermediates on scratch (purged after inactivity: copy what you keep)
OUT_DIR = Path(_env("OUT_DIR", "/N/scratch/gpanayio/scisci-temporal"))
# small results and figures on slate
RESULTS_DIR = Path(_env("RESULTS_DIR", str(BASE / "results")))

# Output of step 0, input of step 1
PAPERS_TSV = OUT_DIR / "papers.tsv"
CITES_NPZ = OUT_DIR / "paper_cites.npz"

# Optional: static roles; its primary_discipline overrides the computed home
ROLES_CSV = _env("ROLES_CSV", None)  # e.g. "/N/slate/gpanayio/scisci-roles/obj/brokerage_roles.csv"

# ── RAW TSV LAYOUT (0-based, from the header) ──────────────────────────────────
COL_ID = 0                 # id
COL_TYPE = 2               # type
COL_YEAR = 3               # publication_year
COL_AUTHORS = 11           # authorships:author:id   "[1,2,3]"
COL_CONCEPT_IDS = 21       # concepts:id             "[41008148,...]"
COL_CONCEPT_SCORES = 22    # concepts:score          "[0.31,...]"
COL_REFS = 23              # referenced_works        "[...]"
ONLY_ARTICLES = True       # keep type == "article" (as in extract_works.py)
COUNT_CITATIONS = True     # second pass over the TSV for cited-by counts

DISCIPLINES = {            # level-0 concept id -> label (same as filter-papers.py)
    "41008148": "CS",
    "33923547": "Math",
    "121332964": "Physics",
    "86803240": "Biology",
    "144024400": "Sociology",
    "162324750": "Economics",
    "204787440": "Linguistics",
}
MIN_CONCEPT_SCORE = 0.3    # match concept_score_threshold in filter-papers.py

# step-1 reader (papers.tsv written by step 0)
COL_PAPER, COL_YEAR_T, COL_AUTHORS_T, COL_FIELDS_T = "paper_id", "year", "author_ids", "fields"
LIST_SEP = ";"
CHUNKSIZE = 2_000_000

# ── POPULATION ─────────────────────────────────────────────────────────────────
YEAR_MIN, YEAR_MAX = 1990, 2022
MAX_TEAM = 30
MIN_PAPERS_TOTAL = 10      # papers in the whole database (7 disciplines, 1990–2022)
MIN_COAUTHORS_TOTAL = 5
FOCUS_HOMES = ["CS"]       # build networks around these home disciplines (None = everyone)
EXPAND_NEIGHBOURS = True   # also give window scores to population co-authors of the focus
                           # authors (needed for Option B exposure: is my co-author a bridge?)

# ── WINDOWS (one reference year T0 per year; IW/AW slide by 1 year) ────────────
WINDOW = 3                 # IW = [T0-3, T0), AW = [T0, T0+3)
T0_MIN, T0_MAX = 1995, 2018
MIN_AW_PAPERS = 1

# ── OPTION A: field entry ──────────────────────────────────────────────────────
CS_FIELD = "CS"
X_FIELDS = ["Math", "Physics", "Biology", "Sociology", "Economics", "Linguistics"]
DIRECTIONS = ("cs_to_x",)  # add "x_to_cs" only if X is also in FOCUS_HOMES
MIN_FIELD_PAPERS_PER_WINDOW = int(_env("MIN_FIELD_PAPERS_PER_WINDOW", 3000))  # as in Venturini et al.

# ── OPTION B: role activation ──────────────────────────────────────────────────
MIN_COLLABORATORS_WINDOW = 5
BLAU_WEIGHTED = False
ACTIVE_Q = 0.80            # bridge  : above this within-window, within-discipline quantile
INACTIVE_Q = 0.60          # non-bridge: at or below this (gap = hysteresis band)
BRIDGE_DEF = "either"      # "either": high B_collab OR high B_sim (any non-coordinator role)
                           # "collab": high B_collab;  "liaison": both high
CANDIDATE_HOMES_B = ["CS"]

# Similarity layer (required for "either" / "liaison").
# One or more (embeddings .npy, ids .txt) pairs with rows in the same order; the
# per-discipline outputs of merge-embeddings.py can be listed directly. Duplicate
# paper ids across files are dropped. Files are memory-mapped.
EMBEDDING_FILES = [
    # ("/N/slate/gpanayio/.../CS/embeddings.npy", "/N/slate/gpanayio/.../CS/ids.txt"),
]
if _env("EMBEDDING_FILES", None):  # override: "a.npy,a_ids.txt;b.npy,b_ids.txt"
    EMBEDDING_FILES = [tuple(p.split(",")) for p in _env("EMBEDDING_FILES", "").split(";")]
SIM_TOPK = 20
MIN_SIM_NEIGHBORS = 5
SIM_OVERFETCH = 5          # fetch TOPK * this, then drop own / future papers
SIM_EXACT_MAX = 5_000_000  # above this many vectors use a compressed faiss IVF-PQ index
SIM_NPROBE = 32
SIM_THREADS = int(_env("SLURM_CPUS_PER_TASK", "1"))

# ── ANALYSIS ───────────────────────────────────────────────────────────────────
KMAX = 10
MIN_N_K = 20
F_STAR = 0.10
TOP_Q = 0.10
DILUTION_Q = 0.20
N_BOOT = 1000
N_PERM = 200
SEED = 42
EXPOSURE = "k_w"           # "k_w": joint papers with active co-authors; "k_d": distinct ones
