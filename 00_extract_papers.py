"""
Step 0 — Stream the OpenAlex works TSV into a compact papers table.

Pass 1 keeps works that are articles, published YEAR_MIN–YEAR_MAX, with
1..MAX_TEAM authors and at least one of the DISCIPLINES at concept score
>= MIN_CONCEPT_SCORE. Writes PAPERS_TSV:
    paper_id  year  author_ids (a;b;c)  fields (CS;Math)
Pass 2 (COUNT_CITATIONS) counts, for every kept paper, how many works in the
whole TSV reference it. Writes CITES_NPZ (ids, cited_by) as sorted int64 arrays.
"""
from array import array

import numpy as np

import config as C
from common import log

LOG_EVERY = 10_000_000


def inner(cell):
    cell = cell.strip()
    return cell[1:-1] if len(cell) > 2 else ""


def pass1():
    kept = array("q")
    need = max(C.COL_AUTHORS, C.COL_CONCEPT_SCORES)
    n = n_kept = 0
    with open(C.WORKS_FILE) as fin, open(C.PAPERS_TSV, "w") as fout:
        next(fin)
        fout.write(f"{C.COL_PAPER}\t{C.COL_YEAR_T}\t{C.COL_AUTHORS_T}\t{C.COL_FIELDS_T}\n")
        for line in fin:
            n += 1
            if n % LOG_EVERY == 0:
                log(f"  pass 1: {n / 1e6:.0f}M lines, kept {n_kept:,}")
            p = line.rstrip("\n").split("\t")
            if len(p) <= need:
                continue
            if C.ONLY_ARTICLES and p[C.COL_TYPE] != "article":
                continue
            try:
                year = int(p[C.COL_YEAR])
            except ValueError:
                continue
            if not (C.YEAR_MIN <= year <= C.YEAR_MAX):
                continue
            authors = inner(p[C.COL_AUTHORS])
            if not authors:
                continue
            alist = sorted(set(a.strip() for a in authors.split(",") if a.strip()))
            if not (1 <= len(alist) <= C.MAX_TEAM):
                continue
            cids = inner(p[C.COL_CONCEPT_IDS]).split(",")
            scs = inner(p[C.COL_CONCEPT_SCORES]).split(",")
            fields = []
            for cid, sc in zip(cids, scs):
                lab = C.DISCIPLINES.get(cid.strip())
                if lab is not None:
                    try:
                        if float(sc) >= C.MIN_CONCEPT_SCORE:
                            fields.append(lab)
                    except ValueError:
                        pass
            if not fields:
                continue
            try:
                pid = int(p[C.COL_ID])
            except ValueError:
                continue
            fout.write(f"{pid}\t{year}\t{';'.join(alist)}\t{';'.join(sorted(set(fields)))}\n")
            kept.append(pid)
            n_kept += 1
    log(f"  pass 1 done: {n:,} lines, {n_kept:,} papers kept")
    return np.unique(np.frombuffer(kept, dtype=np.int64))


def pass2(ids):
    counts = np.zeros(len(ids), dtype=np.int64)
    buf, n = [], 0

    def flush():
        if not buf:
            return
        refs = np.array(",".join(buf).split(","), dtype=np.int64)
        pos = np.searchsorted(ids, refs)
        pos[pos == len(ids)] = 0
        hit = ids[pos] == refs
        counts[:] += np.bincount(pos[hit], minlength=len(ids))
        buf.clear()

    with open(C.WORKS_FILE) as fin:
        next(fin)
        for line in fin:
            n += 1
            p = line.rstrip("\n").split("\t", C.COL_REFS + 1)
            if len(p) > C.COL_REFS:
                r = inner(p[C.COL_REFS]).replace(" ", "")
                if r:
                    buf.append(r)
            if n % 500_000 == 0:
                flush()
            if n % LOG_EVERY == 0:
                log(f"  pass 2: {n / 1e6:.0f}M lines, {counts.sum():,} citations to kept papers")
    flush()
    log(f"  pass 2 done: {counts.sum():,} citations, {(counts > 0).mean():.1%} of papers cited")
    return counts


def main():
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Streaming {C.WORKS_FILE}")
    ids = pass1()
    if C.COUNT_CITATIONS:
        log("Counting citations")
        np.savez(C.CITES_NPZ, ids=ids, cited_by=pass2(ids))
        log(f"  saved {C.CITES_NPZ}")
    log("Done")


if __name__ == "__main__":
    main()
