"""
Step 2A — Option A: field entry ("topic" = discipline X).

For each focal pair and reference year T0:
  IW = [T0-T, T0), AW = [T0, T0+T)
  active source  : author who has published in the focal field at or before the joint paper
  candidate      : population author whose home is the candidate home, who has never
                   published in the focal field before T0, has >= 1 paper in the IW and
                   >= MIN_AW_PAPERS papers in the AW (so non-activation is observable)
  exposure k_w   : joint IW papers with active co-authors, counted per co-author
                   (Venturini et al., definition 1); k_d = distinct active co-authors
  activation     : first paper in the focal field falls in the AW
  chaperoned     : that first paper is co-authored with one of the IW active contacts

Focal pairs (CS+X): "cs_to_x"  candidates home=CS, focal field=X
                    "x_to_cs"  candidates home=X,  focal field=CS

Outputs: panel_A, contacts_A, sources_A
"""
import numpy as np
import pandas as pd

import config as C
from common import log, load, save, reference_years, in_window


def focal_specs():
    specs = []
    for x in C.X_FIELDS:
        if "cs_to_x" in C.DIRECTIONS:
            specs.append({"focal": f"CS->{x}", "cand_home": C.CS_FIELD, "field": x})
        if "x_to_cs" in C.DIRECTIONS:
            if C.FOCUS_HOMES and x not in C.FOCUS_HOMES:
                log(f"skipping {x}->CS: {x} is not in FOCUS_HOMES, so its authors were "
                    f"not all reconstructed (only those who co-author with CS)")
                continue
            specs.append({"focal": f"{x}->CS", "cand_home": x, "field": C.CS_FIELD})
    return specs


def main():
    papers = load("papers")
    pf = load("paper_fields")
    events = load("events")
    pairs = load("pairs")
    home = load("home").set_index("author").home
    afy = load("author_field_year")
    popdf = load("population").set_index("author")

    t0s = reference_years(papers.year.min(), papers.year.max())
    log(f"Reference years: {t0s[0]}–{t0s[-1]} (T = {C.WINDOW})")

    events["home"] = events.author.map(home)
    pyear = papers.set_index("paper").year
    n_ev_by_year = events.groupby(["author", "year"]).size().rename("n").reset_index()

    panels, contacts_all, sources_all = [], [], []
    for spec in focal_specs():
        X, focal = spec["field"], spec["focal"]
        xpapers = pf.loc[pf.field == X, "paper"]
        if xpapers.empty:
            log(f"[{focal}] field '{X}' not found in paper_fields — check X_FIELDS labels; skipping")
            continue
        x_year_counts = pyear.reindex(xpapers).value_counts()
        xset = set(xpapers)
        afx = afy[afy.field == X]
        first_x = afx.groupby("author").year.min()

        cand_pool = popdf.index[(popdf.index.map(home) == spec["cand_home"])]
        cand_pool = cand_pool[cand_pool.map(home) != X]
        log(f"[{focal}] candidate pool (home={spec['cand_home']}): {len(cand_pool):,}")

        p_pool = pairs[pairs.a.isin(set(cand_pool))].copy()
        p_pool["b_first"] = p_pool.b.map(first_x)
        ev_pool = n_ev_by_year[n_ev_by_year.author.isin(set(cand_pool))]
        ev_x_pool = events.loc[events.author.isin(set(cand_pool)) & events.paper.isin(xset),
                               ["author", "paper", "year"]]

        for T0 in t0s:
            iw, aw = T0 - C.WINDOW, T0
            n_iw_x = x_year_counts[(x_year_counts.index >= iw) & (x_year_counts.index < iw + C.WINDOW)].sum()
            n_aw_x = x_year_counts[(x_year_counts.index >= aw) & (x_year_counts.index < aw + C.WINDOW)].sum()
            if min(n_iw_x, n_aw_x) < C.MIN_FIELD_PAPERS_PER_WINDOW:
                continue

            # candidates: never in X before T0, active in IW, can be observed in AW
            fx = first_x.reindex(cand_pool)
            never_before = fx.isna() | (fx >= T0)
            n_iw = ev_pool[in_window(ev_pool.year, iw)].groupby("author").n.sum()
            n_aw = ev_pool[in_window(ev_pool.year, aw)].groupby("author").n.sum()
            cand = cand_pool[never_before.values]
            cand = cand[(n_iw.reindex(cand).fillna(0) >= 1).values &
                        (n_aw.reindex(cand).fillna(0) >= C.MIN_AW_PAPERS).values]
            if len(cand) == 0:
                continue
            cset = set(cand)

            # IW collaboration of candidates
            piw = p_pool[p_pool.a.isin(cset) & in_window(p_pool.year, iw)]
            deg = piw.groupby("a").b.nunique()
            team = piw.drop_duplicates(["a", "paper"]).groupby("a").n_authors.mean()

            # exposure: co-author already active in X at the time of the joint paper
            act = piw[piw.b_first <= piw.year]
            contacts = act.groupby(["a", "b"]).size().rename("w").reset_index()
            k_w = contacts.groupby("a").w.sum()
            k_d = contacts.groupby("a").b.size()
            k_pap = act.groupby("a").paper.nunique()

            # activation in AW
            fxc = first_x.reindex(cand)
            activated = (fxc >= aw) & (fxc < aw + C.WINDOW)

            # chaperoning: first X paper co-authored with an IW active contact
            act_ids = set(fxc.index[activated.values])
            chap = pd.Series(False, index=cand)
            chap_pairs = pd.DataFrame(columns=["a", "b"])
            if act_ids:
                fp = ev_x_pool[ev_x_pool.author.isin(act_ids)]
                fp = fp[fp.year == fp.author.map(first_x)]
                fco = p_pool[p_pool.a.isin(act_ids) & p_pool.paper.isin(set(fp.paper))].merge(
                    fp[["author", "paper"]].rename(columns={"author": "a"}), on=["a", "paper"])
                chap_pairs = fco[["a", "b"]].drop_duplicates().merge(contacts[["a", "b"]], on=["a", "b"])
                chap.loc[chap.index.isin(set(chap_pairs.a))] = True

            panel = pd.DataFrame({
                "candidate": cand,
                "home": spec["cand_home"],
                "k_w": k_w.reindex(cand).fillna(0).astype(int).values,
                "k_d": k_d.reindex(cand).fillna(0).astype(int).values,
                "k_papers": k_pap.reindex(cand).fillna(0).astype(int).values,
                "activated": activated.values,
                "chaperoned": chap.values,
                "career_age": (T0 - popdf.first_year.reindex(cand)).values,
                "n_papers_iw": n_iw.reindex(cand).fillna(0).astype(int).values,
                "degree_iw": deg.reindex(cand).fillna(0).astype(int).values,
                "mean_team_iw": team.reindex(cand).values,
            })
            panel["focal"], panel["T0"] = focal, T0
            panels.append(panel)

            contacts = contacts.rename(columns={"a": "candidate", "b": "source"})
            contacts["focal"], contacts["T0"] = focal, T0
            contacts_all.append(contacts)

            # sources: active in X during the IW (as in the paper), with prominence
            src_iw = afx[in_window(afx.year, iw)].groupby("author").agg(
                productivity=("n_papers", "sum"), cites=("cites", "sum"))
            src = src_iw.loc[src_iw.index.isin(set(contacts.source))].copy()
            src["impact"] = src.cites / src.productivity
            excl = contacts[contacts.candidate.map(k_d) == 1]           # exclusive inactive co-authors
            excl = excl.merge(panel[["candidate", "activated"]], on="candidate")
            excl["chap"] = excl.set_index(["candidate", "source"]).index.isin(
                chap_pairs.set_index(["a", "b"]).index) if len(chap_pairs) else False
            g = excl.groupby("source")
            src["n_excl"] = g.size()
            src["m_excl"] = g.activated.sum()
            src["i_excl"] = g.chap.sum()
            ex_team = piw.merge(excl[["candidate", "source"]].rename(columns={"candidate": "a", "source": "b"}),
                                on=["a", "b"]).groupby("b").n_authors.mean()
            src["mean_team_excl"] = ex_team
            src = src.fillna({"n_excl": 0, "m_excl": 0, "i_excl": 0}).reset_index().rename(
                columns={"author": "source"})
            src["home"] = src.source.map(home)
            src["focal"], src["T0"] = focal, T0
            sources_all.append(src.drop(columns="cites"))

            log(f"[{focal}] T0={T0}: {len(cand):,} candidates, "
                f"{activated.mean():.3f} activated, {len(src):,} sources")

    if not panels:
        raise SystemExit("No panel rows — check X_FIELDS labels and MIN_FIELD_PAPERS_PER_WINDOW")
    save(pd.concat(panels, ignore_index=True), "panel_A")
    save(pd.concat(contacts_all, ignore_index=True), "contacts_A")
    save(pd.concat(sources_all, ignore_index=True), "sources_A")


if __name__ == "__main__":
    main()
