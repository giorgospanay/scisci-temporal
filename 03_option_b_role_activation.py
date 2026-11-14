"""
Step 2B — Option B: role activation (becoming a bridge).

1. Windowed scores: for every window start s, compute per population author
     B_collab(s) = Blau index over the home disciplines of co-authors in [s, s+T)
   (valid only with >= MIN_COLLABORATORS_WINDOW distinct co-authors), and optionally
     B_sim(s)    = Blau index over the fields of the SIM_TOPK papers nearest to the
                   author's window centroid, queried against ONE fixed embedding index
                   (own papers and papers published after the window are dropped).
   States are relative to peers: within-window, within-home-discipline quantiles.
     high  > ACTIVE_Q quantile          low  <= INACTIVE_Q quantile          mid in between

2. Panel per T0 (IW = window starting T0-T, AW = window starting T0):
     BRIDGE_DEF = "collab":  bridge = B_collab high;  non-bridge = B_collab low
     BRIDGE_DEF = "liaison": bridge = both high;      non-bridge = at least one layer low
     BRIDGE_DEF = "either":  bridge = either high;    non-bridge = both layers low (coordinator)
   candidate  : non-bridge in the IW with a valid score in the AW (meets the floor there)
   exposure   : joint IW papers with co-authors who are bridges in the IW
   activation : bridge in the AW (mid-band authors are dropped by the hysteresis rule)

Outputs: window_scores_B, panel_B, contacts_B, sources_B
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from common import log, load, save, reference_years, in_window, blau_from_counts


# ── similarity layer ─────────────────────────────────────────────────────────
def _read_ids(path):
    """Paper ids as int64: an .npy of ints, or text lines like 123, W123, https://openalex.org/W123."""
    if str(path).endswith(".npy"):
        return np.load(path).astype(np.int64)
    s = pd.read_csv(path, header=None, names=["id"], dtype=str).id.str.strip()
    return s.str.extract(r"(\d+)$", expand=False).astype(np.int64).values


class Embeddings:
    """Memory-mapped paper embeddings spread over one or more files."""

    def __init__(self, files):
        self.mats, file_of, local, ids = [], [], [], []
        seen = set()
        for f, (npy, idf) in enumerate(files):
            M = np.load(npy, mmap_mode="r")
            pid = _read_ids(idf)
            if len(pid) != len(M):
                raise ValueError(f"{idf} has {len(pid)} ids but {npy} has {len(M)} rows")
            keep = ~pd.Series(pid).isin(seen).values
            keep &= ~pd.Series(pid).duplicated().values
            seen.update(pid[keep])
            self.mats.append(M)
            file_of.append(np.full(keep.sum(), f, dtype=np.int16))
            local.append(np.flatnonzero(keep))
            ids.append(pid[keep])
            log(f"  {npy}: {len(M):,} rows, {keep.sum():,} kept")
        self.file_of = np.concatenate(file_of)
        self.local = np.concatenate(local)
        self.ids = np.concatenate(ids)
        self.dim = self.mats[0].shape[1]
        self.n = len(self.ids)

    def get(self, rows):
        rows = np.asarray(rows)
        out = np.empty((len(rows), self.dim), dtype=np.float32)
        for f, M in enumerate(self.mats):
            sel = np.flatnonzero(self.file_of[rows] == f)
            if len(sel):
                loc = self.local[rows[sel]]
                order = np.argsort(loc)          # sequential mmap reads
                out[sel[order]] = M[loc[order]]
        return out


def _normalize(V):
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    return V


def build_index(E):
    add_batch = 1_000_000
    try:
        import faiss
        faiss.omp_set_num_threads(max(1, C.SIM_THREADS))
    except ImportError:
        if E.n > C.SIM_EXACT_MAX:
            raise SystemExit("faiss is required for this many embeddings: pip install --user faiss-cpu")
        X = _normalize(E.get(np.arange(E.n)))
        batch = max(1, int(2e8 // E.n))
        log("  similarity index: numpy brute force")

        def search(Q, k):
            out = np.empty((len(Q), k), dtype=np.int64)
            for i in range(0, len(Q), batch):
                S = Q[i:i + batch] @ X.T
                idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
                order = np.argsort(-np.take_along_axis(S, idx, axis=1), axis=1)
                out[i:i + batch] = np.take_along_axis(idx, order, axis=1)
            return out
        return search

    if E.n <= C.SIM_EXACT_MAX:
        index = faiss.IndexFlatIP(E.dim)
        log(f"  similarity index: faiss exact, {E.n:,} vectors")
    else:
        nlist = int(4 * np.sqrt(E.n))
        m = next(m for m in (64, 48, 32, 16, 8) if E.dim % m == 0)
        index = faiss.IndexIVFPQ(faiss.IndexFlatIP(E.dim), E.dim, nlist, m, 8,
                                 faiss.METRIC_INNER_PRODUCT)
        n_train = min(E.n, max(50 * nlist, 500_000))
        rng = np.random.default_rng(C.SEED)
        tr = np.sort(rng.choice(E.n, n_train, replace=False))
        log(f"  similarity index: faiss IVF-PQ (nlist={nlist}, m={m}), training on {n_train:,}")
        index.train(_normalize(E.get(tr)))
        index.nprobe = C.SIM_NPROBE
    for i in range(0, E.n, add_batch):
        index.add(_normalize(E.get(np.arange(i, min(E.n, i + add_batch)))))
        log(f"    added {min(E.n, i + add_batch):,} / {E.n:,}")
    return lambda Q, k: index.search(Q, k)[1]


def windowed_blau_sim(events, pf, papers, starts, wanted):
    """B_sim per (author, window start), only for author-windows in `wanted`."""
    E = Embeddings(C.EMBEDDING_FILES)
    search = build_index(E)
    row_of = pd.Series(np.arange(E.n), index=E.ids)
    years = papers.set_index("paper").year.reindex(E.ids).values.astype(float)
    nf = pf.groupby("paper").size()
    pf_w = pf.assign(w=1.0 / pf.paper.map(nf))
    k_fetch = min(C.SIM_TOPK * C.SIM_OVERFETCH, E.n)

    ev = events[["author", "paper", "year"]].copy()
    ev["row"] = ev.paper.map(row_of)
    ev = ev.dropna(subset=["row"])
    ev["row"] = ev.row.astype(np.int64)
    log(f"  {ev.paper.nunique():,} of the population's papers have embeddings")
    QB = 200_000                                        # authors per query batch
    out = []
    for s in starts:
        want = set(wanted.loc[wanted.s == s, "author"])
        w = ev[in_window(ev.year, s) & ev.author.isin(want)]
        if w.empty:
            continue
        authors = w.author.unique()
        res = []
        for i in range(0, len(authors), QB):
            ab = authors[i:i + QB]
            wb = w[w.author.isin(ab)]
            code = pd.Categorical(wb.author, categories=ab).codes
            V = E.get(wb.row.values)
            Q = np.zeros((len(ab), E.dim), dtype=np.float32)
            np.add.at(Q, code, V)
            nbr = search(_normalize(Q), k_fetch)
            cand = pd.DataFrame({"author": np.repeat(ab, k_fetch), "row": nbr.ravel(),
                                 "rank": np.tile(np.arange(k_fetch), len(ab))})
            cand = cand[cand.row >= 0]
            cand = cand[years[cand.row.values] < s + C.WINDOW]                  # no future papers
            own = ev.loc[ev.author.isin(ab), ["author", "row"]].assign(own=True)
            cand = cand.merge(own, on=["author", "row"], how="left")
            cand = cand[cand.own.isna()].sort_values(["author", "rank"])      # not their own papers
            cand = cand.groupby("author").head(C.SIM_TOPK)
            ok = cand.groupby("author").size()
            cand = cand[cand.author.isin(ok.index[ok >= C.MIN_SIM_NEIGHBORS])]
            cand["paper"] = E.ids[cand.row.values]
            res.append(cand[["author", "paper"]])
        nb = pd.concat(res).merge(pf_w, on="paper")
        b = blau_from_counts(nb, "author", "field", weight_col="w").rename("blau_sim").reset_index()
        b["s"] = s
        out.append(b)
        log(f"  B_sim window {s}: {len(b):,} authors")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["author", "blau_sim", "s"])


# ── windowed collaboration scores ─────────────────────────────────────────────
def windowed_blau_collab(pairs, home, starts):
    p = pairs.assign(b_home=pairs.b.map(home)).dropna(subset=["b_home"])
    out = []
    for s in starts:
        w = p[in_window(p.year, s)]
        n_alt = w.groupby("a").b.nunique()
        if not C.BLAU_WEIGHTED:
            w = w.drop_duplicates(["a", "b"])
        b = blau_from_counts(w, "a", "b_home").rename("blau_collab")
        df = pd.DataFrame({"blau_collab": b, "n_alters": n_alt}).reset_index().rename(columns={"index": "author", "a": "author"})
        df = df[df.n_alters >= C.MIN_COLLABORATORS_WINDOW]
        df["s"] = s
        out.append(df)
    return pd.concat(out, ignore_index=True)


def add_states(df, col, home):
    df["home"] = df.author.map(home)
    g = df.groupby(["s", "home"])[col]
    hi = g.transform(lambda x: x.quantile(C.ACTIVE_Q))
    lo = g.transform(lambda x: x.quantile(C.INACTIVE_Q))
    st = np.where(df[col] > hi, "high", np.where(df[col] <= lo, "low", "mid"))
    df[col.replace("blau_", "") + "_state"] = np.where(df[col].isna(), None, st)
    return df


def main():
    papers = load("papers")
    pairs = load("pairs")
    events = load("events")
    home = load("home").set_index("author").home
    popdf = load("population").set_index("author")

    t0s = reference_years(papers.year.min(), papers.year.max())
    starts = sorted({t - C.WINDOW for t in t0s} | set(t0s))
    log(f"Reference years {t0s[0]}–{t0s[-1]}, {len(starts)} window starts")

    log("Computing windowed B_collab")
    ws = windowed_blau_collab(pairs, home, starts)
    ws = add_states(ws, "blau_collab", home)

    use_sim = bool(C.EMBEDDING_FILES) and all(Path(f[1]).exists() for f in C.EMBEDDING_FILES)  # ids file = complete
    if C.BRIDGE_DEF in ("liaison", "either") and not use_sim:
        raise SystemExit(f"BRIDGE_DEF={C.BRIDGE_DEF!r} needs embeddings: run step 0b "
                         f"(00b_collect_embeddings.py) or set EMBEDDING_FILES")
    if use_sim:
        log("Computing windowed B_sim against a fixed embedding index")
        bs = windowed_blau_sim(events, load("paper_fields"), papers, starts, ws[["author", "s"]])
        bs = add_states(bs, "blau_sim", home).drop(columns="home")
        ws = ws.merge(bs, on=["author", "s"], how="left")
        # role labels at the ACTIVE_Q cut, as in the static typology
        c_hi, s_hi = ws.collab_state == "high", ws.sim_state == "high"
        ws["role"] = np.select([c_hi & s_hi, c_hi & ~s_hi, ~c_hi & s_hi], ["liaison", "gatekeeper", "representative"],
                               default="coordinator")
        ws.loc[ws.sim_state.isna(), "role"] = None

    if C.BRIDGE_DEF == "collab":
        ws["bridge"] = ws.collab_state == "high"
        ws["nonbridge"] = ws.collab_state == "low"
    elif C.BRIDGE_DEF == "liaison":
        ok = ws.sim_state.notna()
        ws["bridge"] = ok & (ws.collab_state == "high") & (ws.sim_state == "high")
        ws["nonbridge"] = ok & ((ws.collab_state == "low") | (ws.sim_state == "low"))
    elif C.BRIDGE_DEF == "either":
        # bridge = gatekeeper, representative or liaison; non-bridge = coordinator
        # clearly below the cut on BOTH layers (mid-band on either layer is unresolved)
        ok = ws.sim_state.notna()
        ws["bridge"] = ok & ((ws.collab_state == "high") | (ws.sim_state == "high"))
        ws["nonbridge"] = ok & (ws.collab_state == "low") & (ws.sim_state == "low")
    else:
        raise SystemExit(f"Unknown BRIDGE_DEF {C.BRIDGE_DEF!r}")
    save(ws, "window_scores_B")

    n_iw_all = events.groupby(["author", "year"]).size().rename("n").reset_index()
    cites_iw = events[["author", "year", "cites"]]
    panels, contacts_all, sources_all = [], [], []
    for T0 in t0s:
        iw, aw = T0 - C.WINDOW, T0
        w_iw = ws[ws.s == iw].set_index("author")
        w_aw = ws[ws.s == aw].set_index("author")
        valid_aw = w_aw.index if C.BRIDGE_DEF == "collab" else w_aw.index[w_aw.sim_state.notna()]

        cand = w_iw.index[w_iw.nonbridge]
        cand = cand[cand.isin(valid_aw)]
        if C.CANDIDATE_HOMES_B:
            cand = cand[cand.map(home).isin(C.CANDIDATE_HOMES_B)]
        if len(cand) == 0:
            continue
        cset = set(cand)
        bridges = set(w_iw.index[w_iw.bridge])

        piw = pairs[pairs.a.isin(cset) & in_window(pairs.year, iw)]
        deg = piw.groupby("a").b.nunique()
        team = piw.drop_duplicates(["a", "paper"]).groupby("a").n_authors.mean()
        act = piw[piw.b.isin(bridges)]
        contacts = act.groupby(["a", "b"]).size().rename("w").reset_index()
        k_w = contacts.groupby("a").w.sum()
        k_d = contacts.groupby("a").b.size()

        aw_state = w_aw.reindex(cand)
        activated = aw_state.bridge.fillna(False).astype(bool)
        # hysteresis: drop candidates that land in the middle band in the AW
        resolved = activated | aw_state.nonbridge.fillna(False).astype(bool)

        n_iw = n_iw_all[in_window(n_iw_all.year, iw)].groupby("author").n.sum()
        panel = pd.DataFrame({
            "candidate": cand,
            "home": cand.map(home),
            "k_w": k_w.reindex(cand).fillna(0).astype(int).values,
            "k_d": k_d.reindex(cand).fillna(0).astype(int).values,
            "activated": activated.values,
            "resolved": resolved.values,
            "chaperoned": np.nan,
            "career_age": (T0 - popdf.first_year.reindex(cand)).values,
            "n_papers_iw": n_iw.reindex(cand).fillna(0).astype(int).values,
            "degree_iw": deg.reindex(cand).fillna(0).astype(int).values,
            "mean_team_iw": team.reindex(cand).values,
            "blau_collab_iw": w_iw.blau_collab.reindex(cand).values,
        })
        if use_sim:
            # which kinds of IW co-authors surround the candidate (all co-authors, not only
            # the active ones), so the regression can compare roles of neighbours
            nb = piw.drop_duplicates(["a", "b"])
            nb_role = nb.b.map(w_iw.role)
            for r in ("liaison", "gatekeeper", "representative"):
                kr = nb[nb_role == r].groupby("a").b.size()
                panel[f"k_d_{r}"] = kr.reindex(cand).fillna(0).astype(int).values
        panel = panel[panel.resolved].drop(columns="resolved")
        panel["focal"], panel["T0"] = f"bridge:{C.BRIDGE_DEF}", T0
        panels.append(panel)

        contacts = contacts[contacts.a.isin(set(panel.candidate))]
        contacts = contacts.rename(columns={"a": "candidate", "b": "source"})
        contacts["focal"], contacts["T0"] = panel.focal.iat[0], T0
        contacts_all.append(contacts)

        # sources: IW bridges in contact with candidates; prominence over IW papers
        src_ids = set(contacts.source)
        ce = cites_iw[cites_iw.author.isin(src_ids) & in_window(cites_iw.year, iw)]
        src = ce.groupby("author").agg(productivity=("cites", "size"), impact=("cites", "mean"))
        kd_panel = panel.set_index("candidate").k_d
        excl = contacts[contacts.candidate.map(kd_panel) == 1].merge(
            panel[["candidate", "activated"]], on="candidate")
        g = excl.groupby("source")
        src["n_excl"] = g.size()
        src["m_excl"] = g.activated.sum()
        src["i_excl"] = np.nan
        src["mean_team_excl"] = piw.merge(
            excl[["candidate", "source"]].rename(columns={"candidate": "a", "source": "b"}),
            on=["a", "b"]).groupby("b").n_authors.mean()
        src = src.fillna({"n_excl": 0, "m_excl": 0}).reset_index().rename(columns={"author": "source"})
        src["home"] = src.source.map(home)
        if use_sim:
            src["role"] = src.source.map(w_iw.role)
        src["focal"], src["T0"] = panel.focal.iat[0], T0
        sources_all.append(src)

        log(f"T0={T0}: {len(panel):,} candidates, {panel.activated.mean():.3f} activated, "
            f"{len(src):,} bridge sources")

    if not panels:
        raise SystemExit("No panel rows — try lowering MIN_COLLABORATORS_WINDOW or widening WINDOW")
    save(pd.concat(panels, ignore_index=True), "panel_B")
    save(pd.concat(contacts_all, ignore_index=True), "contacts_B")
    save(pd.concat(sources_all, ignore_index=True), "sources_B")


if __name__ == "__main__":
    main()
