"""
Step 1 — Build the event tables from papers.tsv (step 0).

Population: authors with >= MIN_PAPERS_TOTAL papers and >= MIN_COAUTHORS_TOTAL
distinct co-authors. Networks are built around FOCUS_HOMES authors (and, with
EXPAND_NEIGHBOURS, their population co-authors), so Option B can tell whether
a co-author is a bridge.

Outputs (OUT_DIR):
  papers         paper, year, n_authors, cites
  paper_fields   paper, field
  events         author, paper, year, n_authors, cites      (population only)
  pairs          a, b, paper, year, n_authors               (a in population; one row per joint paper)
  home           author, home                               (every author in the data)
  author_field_year  author, field, year, n_papers, cites   (population and their co-authors)
  population     author, first_year, n_papers, n_coauthors, home

All ids are int64. The event list keeps one row per co-authored paper, so any
window is just a year filter.
"""
import numpy as np
import pandas as pd

import config as C
from common import log, save

PAIR_BATCH = 2_000_000   # papers per self-join batch


def read_papers():
    papers, ap, pf = [], [], []
    reader = pd.read_csv(C.PAPERS_TSV, sep="\t", chunksize=C.CHUNKSIZE,
                         dtype={C.COL_PAPER: np.int64, C.COL_YEAR_T: np.int16,
                                C.COL_AUTHORS_T: str, C.COL_FIELDS_T: str})
    for i, ch in enumerate(reader):
        ch = ch[ch[C.COL_YEAR_T].between(C.YEAR_MIN, C.YEAR_MAX)]
        a = ch[[C.COL_PAPER]].assign(author=ch[C.COL_AUTHORS_T].str.split(C.LIST_SEP)).explode("author")
        a = a.rename(columns={C.COL_PAPER: "paper"})
        a["author"] = a["author"].astype(np.int64)
        a = a.drop_duplicates()
        n = a.groupby("paper").size().rename("n_authors").astype(np.int16)
        keep = n.index[n <= C.MAX_TEAM]
        ap.append(a[a.paper.isin(keep)])
        pp = ch.set_index(C.COL_PAPER).loc[keep, [C.COL_YEAR_T]].rename(columns={C.COL_YEAR_T: "year"})
        pp["n_authors"] = n.loc[keep]
        papers.append(pp.reset_index().rename(columns={C.COL_PAPER: "paper"}))
        f = ch[[C.COL_PAPER]].assign(field=ch[C.COL_FIELDS_T].str.split(C.LIST_SEP)).explode("field")
        f = f.rename(columns={C.COL_PAPER: "paper"})
        pf.append(f[f.paper.isin(keep)])
        log(f"  chunk {i}: {len(keep):,} papers")
    papers = pd.concat(papers, ignore_index=True).drop_duplicates("paper")
    ap = pd.concat(ap, ignore_index=True).drop_duplicates()
    pf = pd.concat(pf, ignore_index=True).drop_duplicates()
    pf["field"] = pf["field"].astype("category")

    papers["cites"] = 0.0
    if C.CITES_NPZ.exists():
        z = np.load(C.CITES_NPZ)
        cites = pd.Series(z["cited_by"].astype(np.float32), index=z["ids"])
        papers["cites"] = papers.paper.map(cites).fillna(0).astype(np.float32).values
        log(f"  citation counts attached from {C.CITES_NPZ}")
    else:
        log("  no citation counts found — impact analyses will be uninformative")
    return papers, ap, pf


def build_pairs(ap, focal_authors, papers_n):
    """Event list (a, b, paper) for a in focal_authors, in batches of papers."""
    pp = np.unique(ap.loc[ap.author.isin(focal_authors), "paper"].values)
    sub = ap.loc[ap.paper.isin(pp), ["paper", "author"]]
    out = []
    for i in range(0, len(pp), PAIR_BATCH):
        batch = pp[i:i + PAIR_BATCH]
        s = sub[sub.paper.isin(batch)]
        left = s[s.author.isin(focal_authors)]
        m = left.merge(s, on="paper", suffixes=("", "_b"))
        m = m[m.author != m.author_b].rename(columns={"author": "a", "author_b": "b"})
        out.append(m)
        log(f"    pairs batch {i // PAIR_BATCH}: {len(m):,} rows")
    pairs = pd.concat(out, ignore_index=True)
    return pairs.merge(papers_n, on="paper")


def main():
    log(f"Reading {C.PAPERS_TSV}")
    papers, ap, pf = read_papers()
    log(f"  {len(papers):,} papers, {ap.author.nunique():,} authors, "
        f"fields: {sorted(pf.field.cat.categories)}")
    ap = ap.merge(papers[["paper", "year"]], on="paper")

    # home discipline of every author: fractional paper counts per field
    log("Home disciplines")
    nf = pf.groupby("paper").size().rename("nf")
    af = ap[["author", "paper", "year"]].merge(pf, on="paper")
    af["w"] = (1.0 / af.paper.map(nf)).astype(np.float32)
    home = (af.groupby(["author", "field"], observed=True).w.sum().reset_index()
              .sort_values(["author", "w"], ascending=[True, False])
              .drop_duplicates("author")[["author", "field"]]
              .rename(columns={"field": "home"}))
    home["home"] = home["home"].astype(str)
    if C.ROLES_CSV:
        roles = pd.read_csv(C.ROLES_CSV, usecols=["author_id", "primary_discipline"]).dropna()
        roles = roles.set_index(roles.author_id.astype(np.int64)).primary_discipline
        override = home.author.map(roles)
        log(f"  home overridden from {C.ROLES_CSV} for {override.notna().sum():,} authors")
        home["home"] = override.fillna(home["home"])
    home_s = home.set_index("author").home

    # population, stage 1 (paper floor) and focus
    n_pap = ap.groupby("author").size()
    pop1 = pd.Index(n_pap.index[n_pap >= C.MIN_PAPERS_TOTAL])
    log(f"  {len(pop1):,} authors with >= {C.MIN_PAPERS_TOTAL} papers")
    if C.FOCUS_HOMES:
        focus = pop1[pop1.map(home_s).isin(C.FOCUS_HOMES)]
        log(f"  {len(focus):,} of them with home in {C.FOCUS_HOMES}")
    else:
        focus = pop1
    papers_n = papers[["paper", "year", "n_authors"]]

    # neighbours of the focus who are themselves in the paper-floor population
    scope = set(focus)
    if C.FOCUS_HOMES and C.EXPAND_NEIGHBOURS:
        fp = np.unique(ap.loc[ap.author.isin(scope), "paper"].values)
        nbrs = set(ap.loc[ap.paper.isin(fp), "author"].unique()) & set(pop1)
        log(f"  + {len(nbrs - scope):,} population co-authors of the focus")
        scope |= nbrs

    log("Building co-authorship event list")
    pairs = build_pairs(ap, scope, papers_n)
    n_co = pairs.groupby("a").b.nunique()
    pop = set(n_co.index[n_co >= C.MIN_COAUTHORS_TOTAL])
    pairs = pairs[pairs.a.isin(pop)].reset_index(drop=True)
    for c, t in (("year", np.int16), ("n_authors", np.int16)):
        pairs[c] = pairs[c].astype(t)
    log(f"  population after co-author floor: {len(pop):,}; {len(pairs):,} pair-paper rows")

    needed = pop | set(pairs.b.unique())
    afy = (af[af.author.isin(needed)].merge(papers[["paper", "cites"]], on="paper")
             .groupby(["author", "field", "year"], observed=True)
             .agg(n_papers=("paper", "size"), cites=("cites", "sum")).reset_index())
    afy["field"] = afy["field"].astype(str)
    del af

    events = ap[ap.author.isin(pop)].merge(papers[["paper", "n_authors", "cites"]], on="paper")
    popdf = (events.groupby("author").agg(first_year=("year", "min"), n_papers=("paper", "size"))
                   .join(n_co.rename("n_coauthors")).reset_index())
    popdf["home"] = popdf.author.map(home_s)
    log("Population by home discipline:\n" + popdf.home.value_counts().to_string())

    pf["field"] = pf["field"].astype(str)
    save(papers, "papers")
    save(pf, "paper_fields")
    save(events, "events")
    save(pairs[["a", "b", "paper", "year", "n_authors"]], "pairs")
    save(home, "home")
    save(afy, "author_field_year")
    save(popdf, "population")
    log("Done")


if __name__ == "__main__":
    main()
