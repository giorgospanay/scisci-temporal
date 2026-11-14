"""
Step 0b — Collect the SPECTER2 chunk files into one array of the papers we need.

Input : EMBEDDING_CHUNK_DIR/embeddings_<k>.npy + ids_<k>.txt  (same row order)
Needed: the papers kept by step 0 (papers.tsv / paper_cites.npz)
Output: EMB_NPY  (n_needed x dim, EMB_DTYPE)
        EMB_IDS  (int64 paper ids, same order) — written LAST, so it marks completion

Pass 1 reads the id files (in parallel threads) to find which rows are needed; the
first occurrence wins if a paper appears in several chunks. The selection is saved,
so a restarted job skips pass 1.
Pass 2 copies those rows chunk by chunk into a memory-mapped .npy (parallel reads,
sequential writes) and checkpoints its progress, so a job that hits the time limit
continues where it stopped when resubmitted.
At the end a random sample of rows is compared with the source chunks.

Resuming a run that stopped WITHOUT a checkpoint (the first version of this script):
    START_GROUP=140000 python 00b_collect_embeddings.py
reuses the existing partial array and starts copying at that chunk group (use the
last number printed in the log). The final check reports any rows that do not match.
"""
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from common import log

DIGITS = re.compile(r"(\d+)\s*$")
THREADS = int(os.environ.get("EMB_THREADS", 2 * max(1, C.SIM_THREADS)))
FLUSH_EVERY = 5_000          # chunk groups between checkpoints
SEL = C.OUT_DIR / "embeddings_selection.npz"
PROGRESS = C.OUT_DIR / "embeddings_progress.txt"


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


def pass1(pairs, need):
    def scan(item):
        i, (k, npy, idf) = item
        pid = read_ids(idf)
        shape = np.load(npy, mmap_mode="r").shape
        if shape[0] != len(pid):
            return i, k, None, None, shape
        pos = np.searchsorted(need, pid)
        pos[pos == len(need)] = 0
        hit = np.flatnonzero(need[pos] == pid)
        return i, k, hit, pid[hit], shape

    sel_chunk, sel_row, sel_id = [], [], []
    bad, dim = 0, None
    with ThreadPoolExecutor(THREADS) as ex:
        for j, (i, k, hit, ids, shape) in enumerate(ex.map(scan, enumerate(pairs), chunksize=64)):
            if hit is None:
                bad += 1
                log(f"  skip chunk {k}: {shape[0]} rows vs id count mismatch")
            else:
                dim = dim or shape[1]
                sel_chunk.append(np.full(len(hit), i, dtype=np.int32))
                sel_row.append(hit.astype(np.int32))
                sel_id.append(ids)
            if (j + 1) % 10_000 == 0:
                log(f"  pass 1: {j + 1:,}/{len(pairs):,} chunks")
    sel_chunk, sel_row, sel_id = map(np.concatenate, (sel_chunk, sel_row, sel_id))
    _, first = np.unique(sel_id, return_index=True)          # drop duplicate papers
    first.sort()
    log(f"  {len(first):,} unique papers; {bad} bad chunks")
    return sel_chunk[first], sel_row[first], sel_id[first], dim


def main():
    if C.EMB_IDS.exists():
        log(f"{C.EMB_IDS} exists — embeddings already collected (delete it to redo)")
        return
    d = Path(C.EMBEDDING_CHUNK_DIR)
    pairs = chunk_pairs(d)
    keys = np.array([k for k, _, _ in pairs])
    log(f"{len(pairs):,} chunk pairs in {d}; {THREADS} reader threads")

    if SEL.exists() and np.array_equal(np.load(SEL)["keys"], keys):
        z = np.load(SEL)
        sel_chunk, sel_row, sel_id, dim = z["chunk"], z["row"], z["id"], int(z["dim"])
        log(f"pass 1 loaded from {SEL}")
    else:
        need = needed_ids()
        log(f"pass 1: {len(need):,} papers needed (from step 0)")
        sel_chunk, sel_row, sel_id, dim = pass1(pairs, need)
        log(f"  coverage: {len(sel_id) / len(need):.1%} of needed papers have embeddings")
        np.savez(SEL, chunk=sel_chunk, row=sel_row, id=sel_id, dim=dim, keys=keys)
    n = len(sel_id)

    starts = np.flatnonzero(np.r_[True, sel_chunk[1:] != sel_chunk[:-1]])
    ends = np.r_[starts[1:], n]
    # the checkpoint wins over START_GROUP (it may have been moved back by the spot check)
    g0 = int(PROGRESS.read_text().strip() or 0) if PROGRESS.exists() else int(os.environ.get("START_GROUP", 0))
    if g0 > 0 and C.EMB_NPY.exists():
        out = np.load(C.EMB_NPY, mmap_mode="r+")
        if out.shape != (n, dim):
            raise SystemExit(f"existing {C.EMB_NPY} has shape {out.shape}, expected {(n, dim)}: delete it")
        log(f"pass 2: resuming at chunk group {g0:,}/{len(starts):,}")
    else:
        g0 = 0
        out = np.lib.format.open_memmap(C.EMB_NPY, mode="w+", dtype=C.EMB_DTYPE, shape=(n, dim))
        log(f"pass 2: {len(starts):,} chunk groups -> {C.EMB_NPY} ({n:,} x {dim})")

    def load(g):
        a, b = starts[g], ends[g]
        M = np.load(pairs[sel_chunk[a]][1])              # whole chunk: one sequential read
        return g, M[sel_row[a:b]]

    with ThreadPoolExecutor(THREADS) as ex:
        for g, rows in ex.map(load, range(g0, len(starts)), chunksize=16):
            out[starts[g]:ends[g]] = rows
            done = g + 1
            if done % FLUSH_EVERY == 0 or done == len(starts):
                out.flush()
                PROGRESS.write_text(str(done))
            if done % 10_000 == 0:
                log(f"  pass 2: {done:,}/{len(starts):,} chunk groups copied")
    out.flush()

    # spot check against the source chunks (catches rows lost in an unflushed earlier run)
    rng = np.random.default_rng(C.SEED)
    check = np.sort(rng.choice(n, min(n, 5000), replace=False))
    g_of = np.searchsorted(starts, check, side="right") - 1
    bad_groups = []
    for r, g in zip(check, g_of):
        src = np.load(pairs[sel_chunk[r]][1], mmap_mode="r")[sel_row[r]]
        if not np.allclose(out[r].astype(np.float32), src, atol=1e-2, rtol=1e-2):
            bad_groups.append(int(g))
    log(f"spot check: {len(bad_groups)} of {len(check)} sampled rows differ from the source")
    if bad_groups:
        PROGRESS.write_text(str(min(bad_groups)))
        raise SystemExit(f"rows differ from chunk group {min(bad_groups):,} on: resubmit this job "
                         f"and it resumes there (checkpoint updated)")

    np.save(C.EMB_IDS, sel_id)                            # completion marker
    PROGRESS.unlink(missing_ok=True)
    log(f"saved {C.EMB_NPY} ({n:,} x {dim}, {C.EMB_DTYPE}) and {C.EMB_IDS}")


if __name__ == "__main__":
    sys.exit(main())
