"""
Step 0b — Collect the SPECTER2 chunk files into one array of the papers we need.

Input : EMBEDDING_CHUNK_DIR/embeddings_<k>.npy + ids_<k>.txt  (same row order)
Needed: the papers kept by step 0 (papers.tsv / paper_cites.npz)
Output: EMB_NPY  (n_needed x dim, EMB_DTYPE)   and   EMB_IDS (int64 paper ids, same order)

Pass 1 reads only the id files to find which rows are needed (first occurrence wins
if a paper appears in several chunks). Pass 2 copies those rows into one memory-mapped
.npy, so RAM stays at one chunk at a time. Chunks whose id file and array disagree in
length are skipped and reported.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from common import log

DIGITS = re.compile(r"(\d+)\s*$")


def needed_ids():
    if C.CITES_NPZ.exists():
        return np.load(C.CITES_NPZ)["ids"]
    ids = pd.read_csv(C.PAPERS_TSV, sep="\t", usecols=[C.COL_PAPER], dtype=np.int64)[C.COL_PAPER]
    return np.unique(ids.values)


def read_ids(path):
    out = []
    with open(path) as fh:
        for ln in fh:
            m = DIGITS.search(ln)
            out.append(int(m.group(1)) if m else -1)
    return np.array(out, dtype=np.int64)


def chunk_pairs(d):
    emb = {int(p.stem.split("_")[1]): p for p in d.glob("embeddings_*.npy")}
    ids = {int(p.stem.split("_")[1]): p for p in d.glob("ids_*.txt")}
    both = sorted(emb.keys() & ids.keys())
    if len(emb) != len(both) or len(ids) != len(both):
        log(f"  WARNING: {len(emb) - len(both)} arrays without ids, {len(ids) - len(both)} ids without arrays")
    return [(k, emb[k], ids[k]) for k in both]


def main():
    d = Path(C.EMBEDDING_CHUNK_DIR)
    need = needed_ids()
    log(f"{len(need):,} papers needed (from step 0)")
    pairs = chunk_pairs(d)
    log(f"{len(pairs):,} chunk pairs in {d}")

    # pass 1: which rows of which chunk
    sel_chunk, sel_row, sel_id = [], [], []
    bad, dim = 0, None
    for i, (k, npy, idf) in enumerate(pairs):
        pid = read_ids(idf)
        shape = np.load(npy, mmap_mode="r").shape
        if shape[0] != len(pid):
            bad += 1
            log(f"  skip chunk {k}: {shape[0]} rows vs {len(pid)} ids")
            continue
        dim = dim or shape[1]
        pos = np.searchsorted(need, pid)
        pos[pos == len(need)] = 0
        hit = np.flatnonzero(need[pos] == pid)
        sel_chunk.append(np.full(len(hit), i, dtype=np.int32))
        sel_row.append(hit.astype(np.int32))
        sel_id.append(pid[hit])
        if (i + 1) % 10_000 == 0:
            log(f"  pass 1: {i + 1:,}/{len(pairs):,} chunks, {sum(map(len, sel_id)):,} rows matched")
    sel_chunk, sel_row, sel_id = map(np.concatenate, (sel_chunk, sel_row, sel_id))
    _, first = np.unique(sel_id, return_index=True)           # drop duplicate papers
    first.sort()
    sel_chunk, sel_row, sel_id = sel_chunk[first], sel_row[first], sel_id[first]
    n = len(sel_id)
    log(f"  {n:,} unique papers with embeddings ({n / len(need):.1%} of needed); {bad} bad chunks")

    # pass 2: copy rows
    C.EMB_NPY.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(C.EMB_NPY, mode="w+", dtype=C.EMB_DTYPE, shape=(n, dim))
    starts = np.flatnonzero(np.r_[True, sel_chunk[1:] != sel_chunk[:-1]])
    ends = np.r_[starts[1:], n]
    for j, (a, b) in enumerate(zip(starts, ends)):
        M = np.load(pairs[sel_chunk[a]][1], mmap_mode="r")
        out[a:b] = M[sel_row[a:b]]
        if (j + 1) % 10_000 == 0:
            log(f"  pass 2: {j + 1:,}/{len(starts):,} chunks copied")
    out.flush()
    del out
    np.save(C.EMB_IDS, sel_id)
    log(f"saved {C.EMB_NPY} ({n:,} x {dim}, {C.EMB_DTYPE}) and {C.EMB_IDS}")


if __name__ == "__main__":
    sys.exit(main())
