"""Shared helpers: logging, storage, windows, Blau index, bootstrap."""
import time
import numpy as np
import pandas as pd

import config as C

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# ── storage (Parquet if pyarrow is installed, pickle otherwise) ───────────────
try:
    import pyarrow  # noqa: F401
    _EXT = ".parquet"
except ImportError:
    _EXT = ".pkl"


def path(name):
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    return C.OUT_DIR / f"{name}{_EXT}"


def save(df, name):
    p = path(name)
    if _EXT == ".parquet":
        df.to_parquet(p, index=False)
    else:
        df.to_pickle(p)
    log(f"  saved {p} ({len(df):,} rows)")


def load(name):
    p = path(name)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run the earlier pipeline step first")
    return pd.read_parquet(p) if _EXT == ".parquet" else pd.read_pickle(p)


# ── windows ───────────────────────────────────────────────────────────────────
def reference_years(year_min, year_max):
    """T0 values whose IW and AW both lie inside the observed years."""
    lo = max(C.T0_MIN, year_min + C.WINDOW)
    hi = min(C.T0_MAX, year_max - C.WINDOW + 1)
    return list(range(lo, hi + 1))


def in_window(years, start):
    return (years >= start) & (years < start + C.WINDOW)


# ── Blau index ────────────────────────────────────────────────────────────────
def blau_from_counts(df, ego_col, cat_col, weight_col=None):
    """1 - sum_i p_i^2 over categories, per ego. df has one row per (ego, item)."""
    if weight_col is None:
        cnt = df.groupby([ego_col, cat_col]).size()
    else:
        cnt = df.groupby([ego_col, cat_col])[weight_col].sum()
    tot = cnt.groupby(level=0).sum()
    sq = (cnt ** 2).groupby(level=0).sum()
    return (1.0 - sq / tot ** 2).rename("blau")


# ── statistics ────────────────────────────────────────────────────────────────
def boot_mean_ci(x, n_boot=None, seed=None, alpha=0.05):
    """Mean and bootstrap CI of the mean (aggregation across reference years)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return x[0], np.nan, np.nan
    rng = np.random.default_rng(C.SEED if seed is None else seed)
    n_boot = n_boot or C.N_BOOT
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return x.mean(), np.quantile(means, alpha / 2), np.quantile(means, 1 - alpha / 2)


def rank_bins(s, q, rng):
    """Top/bottom-q membership with random tie-breaking (many productivity ties)."""
    jitter = pd.Series(rng.random(len(s)), index=s.index)
    order = pd.DataFrame({"v": s, "j": jitter}).sort_values(["v", "j"])
    n = len(order)
    k = max(1, int(round(q * n)))
    bottom = set(order.index[:k])
    top = set(order.index[-k:])
    return top, bottom
