"""
Step 3 — Analysis, shared by both options.

  python 04_analyze.py A      # field entry
  python 04_analyze.py B      # role activation

Per focal (CS->X, X->CS, or bridge), aggregated over reference years with bootstrap CIs:
  1. Target activation C(k) vs the independent-contacts baseline 1-(1-p)^k     (Exp. I)
  2. Prominence: C(k) for candidates exclusively exposed to top vs bottom 10% sources
  3. Source activation C_s(f*) and chaperoning propensity, top vs bottom 10%   (Exp. II)
  4. Dilution: among top-10% sources, high vs low mean team size
  5. Discrete-time logit with author-clustered SEs (exposure x degree = target-side dilution;
     Option B with embeddings also splits exposure by the source's role)
  6. Stratified null: activation permuted within (T0, home, degree quintile, productivity
     quintile); is the k=0 -> k>=1 jump larger than expected from activity alone?

Outputs: results_<opt>/*.csv and *.png
"""
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from common import log, load, boot_mean_ci, rank_bins

warnings.filterwarnings("ignore", category=RuntimeWarning)
KS = np.arange(cfg.KMAX + 1)


# ── 1. C(k) and baseline ──────────────────────────────────────────────────────
def ck_curve(k, y):
    """Cumulative activation C(k) = P(activate | >= k contacts) and independent baseline."""
    k, y = np.asarray(k), np.asarray(y, dtype=float)
    n1 = (k == 1).sum()
    p = y[k == 1].mean() if n1 else np.nan
    pbase = 1 - (1 - p) ** k
    obs, base, n = [], [], []
    for K in KS:
        sel = k >= K
        n.append(sel.sum())
        obs.append(y[sel].mean() if sel.sum() >= cfg.MIN_N_K else np.nan)
        base.append(pbase[sel].mean() if sel.sum() >= cfg.MIN_N_K else np.nan)
    return np.array(obs), np.array(base), np.array(n)


def target_activation(panel):
    rows = []
    for (focal, T0), g in panel.groupby(["focal", "T0"]):
        obs, base, n = ck_curve(g[cfg.EXPOSURE], g.activated)
        for K in KS:
            rows.append((focal, T0, K, obs[K], base[K], n[K]))
    per = pd.DataFrame(rows, columns=["focal", "T0", "k", "C", "C_base", "n"])
    per["diff"] = per.C - per.C_base
    agg = []
    for (focal, K), g in per.groupby(["focal", "k"]):
        m, lo, hi = boot_mean_ci(g.C)
        d, dlo, dhi = boot_mean_ci(g["diff"])
        agg.append((focal, K, m, lo, hi, g.C_base.mean(), d, dlo, dhi, g.C.notna().sum()))
    agg = pd.DataFrame(agg, columns=["focal", "k", "C", "C_lo", "C_hi", "C_base",
                                     "diff", "diff_lo", "diff_hi", "n_T0"])
    return per, agg


def plot_ck(agg, out):
    focals = agg.focal.unique()
    ncol = min(4, len(focals))
    nrow = int(np.ceil(len(focals) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False, sharex=True)
    for ax, f in zip(axes.flat, focals):
        g = agg[agg.focal == f]
        ax.fill_between(g.k, g.C_lo, g.C_hi, color="#6a3d9a", alpha=0.2, lw=0)
        ax.plot(g.k, g.C, "o-", color="#6a3d9a", ms=3, label="observed")
        ax.plot(g.k, g.C_base, "--", color="#33a02c", label="independent baseline")
        ax.set_title(f, fontsize=9)
        ax.set_xlabel(f"exposure {cfg.EXPOSURE} (≥)")
        ax.set_ylabel("C(k)")
    for ax in list(axes.flat)[len(focals):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ── 2. prominence (target side) ───────────────────────────────────────────────
def prominence_target(panel, contacts, sources, metric, kmax=5):
    rng = np.random.default_rng(cfg.SEED)
    rows = []
    for (focal, T0), src in sources.groupby(["focal", "T0"]):
        src = src.dropna(subset=[metric])
        src = src[src.productivity > 0]
        if len(src) < 20:
            continue
        top, bot = rank_bins(src.set_index("source")[metric], cfg.TOP_Q, rng)
        ct = contacts[(contacts.focal == focal) & (contacts.T0 == T0)]
        has_top = ct[ct.source.isin(top)].groupby("candidate").size()
        has_bot = ct[ct.source.isin(bot)].groupby("candidate").size()
        g = panel[(panel.focal == focal) & (panel.T0 == T0)].set_index("candidate")
        ht = g.index.isin(has_top.index)
        hb = g.index.isin(has_bot.index)
        gt, gb = g[ht & ~hb], g[hb & ~ht]
        for K in range(1, kmax + 1):
            a, b = gt[gt[cfg.EXPOSURE] >= K], gb[gb[cfg.EXPOSURE] >= K]
            if len(a) >= cfg.MIN_N_K and len(b) >= cfg.MIN_N_K:
                rows.append((focal, T0, K, a.activated.mean() - b.activated.mean()))
    per = pd.DataFrame(rows, columns=["focal", "T0", "k", "diff_top_minus_bottom"])
    return summarize(per, ["focal", "k"], "diff_top_minus_bottom")


# ── 3–4. source activation, chaperoning, dilution ─────────────────────────────
def source_side(sources, metric):
    rng = np.random.default_rng(cfg.SEED)
    rows = []
    for (focal, T0), src in sources.groupby(["focal", "T0"]):
        src = src[(src.n_excl > 0) & (src.productivity > 0)].dropna(subset=[metric])
        if len(src) < 20:
            continue
        s = src.set_index("source")
        top, bot = rank_bins(s[metric], cfg.TOP_Q, rng)
        ps = s.m_excl / s.n_excl
        cs = lambda ids: (ps.loc[list(ids)] >= cfg.F_STAR).mean()
        row = {"focal": focal, "T0": T0, "source_act": cs(top) - cs(bot)}
        if s.i_excl.notna().any():
            pc = (s.i_excl / s.m_excl).where(s.m_excl > 0)
            pc_t, pc_b = pc.loc[list(top)].dropna(), pc.loc[list(bot)].dropna()
            row["chaperoning"] = ((pc_t >= cfg.F_STAR).mean() - (pc_b >= cfg.F_STAR).mean()
                                  if len(pc_t) and len(pc_b) else np.nan)
        # dilution among the most prominent sources
        st = s.loc[list(top)].dropna(subset=["mean_team_excl"])
        if len(st) >= 10:
            big, small = rank_bins(st.mean_team_excl, cfg.DILUTION_Q, rng)
            row["dilution_big_minus_small"] = cs(big) - cs(small)
        rows.append(row)
    per = pd.DataFrame(rows)
    out = []
    for col in [c for c in ("source_act", "chaperoning", "dilution_big_minus_small") if c in per]:
        s = summarize(per, ["focal"], col)
        s.insert(1, "measure", col)
        out.append(s.rename(columns={col: "mean"}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summarize(per, keys, col):
    rows = []
    for key, g in per.groupby(keys):
        m, lo, hi = boot_mean_ci(g[col])
        rows.append((*np.atleast_1d(key), m, lo, hi, g[col].notna().sum()))
    return pd.DataFrame(rows, columns=keys + [col, "lo", "hi", "n_T0"])


def plot_effects(df, label_col, value_col, title, out):
    if df.empty:
        return
    df = df.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(df) + 1.2))
    y = np.arange(len(df))
    sig = (df.lo > 0) | (df.hi < 0)
    ax.errorbar(df[value_col], y, xerr=[df[value_col] - df.lo, df.hi - df[value_col]],
                fmt="none", ecolor="#999", lw=1)
    ax.scatter(df[value_col], y, c=np.where(sig, "#d62728", "#7f7f7f"), zorder=3, s=18)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df[label_col], fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ── 5. regression ─────────────────────────────────────────────────────────────
def regressions(panel, option):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        log("  statsmodels not installed — skipping regressions")
        return pd.DataFrame()
    rows = []
    role_cols = [c for c in panel.columns if c.startswith("k_d_")]
    for focal, g in panel.groupby("focal"):
        g = g.dropna(subset=["career_age", "mean_team_iw"]).copy()
        if g.activated.sum() < 30:
            continue
        g["y"] = g.activated.astype(int)
        g["lk"] = np.log1p(g[cfg.EXPOSURE])
        g["ldeg"] = np.log1p(g.degree_iw)
        g["lpap"] = np.log1p(g.n_papers_iw)
        rhs = "lk * ldeg + career_age + lpap + mean_team_iw + C(T0)"
        usable = [c for c in role_cols if g[c].nunique() > 1]
        if usable:
            # neighbour roles (coordinators are the reference, absorbed by degree)
            for c in usable:
                g["l" + c] = np.log1p(g[c])
            rhs = " + ".join("l" + c for c in usable) + " + ldeg + career_age + lpap + mean_team_iw + C(T0)"
        if g.home.nunique() > 1:
            rhs += " + C(home)"
        try:
            codes = pd.factorize(g.candidate)[0]
            fit = smf.logit("y ~ " + rhs, data=g).fit(disp=0, cov_type="cluster",
                                                       cov_kwds={"groups": codes})
        except Exception as e:  # separation, singular design, ...
            log(f"  [{focal}] logit failed: {e}")
            continue
        for term in fit.params.index:
            if term.startswith("C(T0)") or term == "Intercept":
                continue
            ci = fit.conf_int().loc[term]
            rows.append((focal, term, fit.params[term], ci[0], ci[1], fit.pvalues[term], int(fit.nobs)))
    return pd.DataFrame(rows, columns=["focal", "term", "coef", "lo", "hi", "p", "n"])


# ── 6. stratified permutation null ────────────────────────────────────────────
def stratified_null(panel):
    rng = np.random.default_rng(cfg.SEED)
    rows = []
    for focal, g in panel.groupby("focal"):
        g = g.copy()
        g["dq"] = g.groupby("T0").degree_iw.transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
        g["pq"] = g.groupby("T0").n_papers_iw.transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
        strata = g.groupby(["T0", "home", "dq", "pq"], dropna=False).ngroup().values
        y = g.activated.values.astype(float)
        exposed = g[cfg.EXPOSURE].values >= 1
        if exposed.sum() < cfg.MIN_N_K or (~exposed).sum() < cfg.MIN_N_K:
            continue
        obs = y[exposed].mean() - y[~exposed].mean()
        order = np.argsort(strata, kind="stable")
        null = np.empty(cfg.N_PERM)
        for i in range(cfg.N_PERM):
            perm = np.lexsort((rng.random(len(y)), strata))
            yp = np.empty_like(y)
            yp[order] = y[perm]
            null[i] = yp[exposed].mean() - yp[~exposed].mean()
        rows.append((focal, obs, null.mean(), np.quantile(null, 0.975),
                     (np.sum(null >= obs) + 1) / (cfg.N_PERM + 1)))
    return pd.DataFrame(rows, columns=["focal", "jump_observed", "jump_null_mean",
                                       "jump_null_q975", "p_one_sided"])


def main():
    opt = (sys.argv[1] if len(sys.argv) > 1 else "A").upper()
    panel, contacts, sources = load(f"panel_{opt}"), load(f"contacts_{opt}"), load(f"sources_{opt}")
    out = cfg.RESULTS_DIR / f"results_{opt}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"Option {opt}: {len(panel):,} candidate-windows, {panel.focal.nunique()} focal set(s)")

    log("1. Target activation C(k)")
    per, agg = target_activation(panel)
    per.to_csv(out / "ck_per_T0.csv", index=False)
    agg.to_csv(out / "ck_summary.csv", index=False)
    plot_ck(agg, out / "ck_curves.png")

    log("2–4. Prominence, source activation, chaperoning, dilution")
    for metric in ("productivity", "impact"):
        pt = prominence_target(panel, contacts, sources, metric)
        pt.to_csv(out / f"prominence_target_{metric}.csv", index=False)
        if not pt.empty:
            pt["label"] = pt.focal + "  k≥" + pt.k.astype(str)
            plot_effects(pt, "label", "diff_top_minus_bottom",
                         f"C(k): exclusive top − bottom 10% sources ({metric})",
                         out / f"prominence_target_{metric}.png")
        ss = source_side(sources, metric)
        ss.to_csv(out / f"source_side_{metric}.csv", index=False)
        if not ss.empty:
            ss["label"] = ss.focal + "  " + ss.measure
            plot_effects(ss, "label", "mean",
                         f"Source side, top − bottom ({metric}); dilution = big − small teams",
                         out / f"source_side_{metric}.png")

    log("5. Regressions")
    reg = regressions(panel, opt)
    reg.to_csv(out / "logit.csv", index=False)

    log("6. Stratified null")
    nul = stratified_null(panel)
    nul.to_csv(out / "null_stratified.csv", index=False)
    if not nul.empty:
        print(nul.round(4).to_string(index=False))
    log(f"Results in {out}")


if __name__ == "__main__":
    main()
