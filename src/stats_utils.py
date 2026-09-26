"""Inference with few clusters (models). Each function takes one value per cluster
(e.g. the seven model-level means for one dimension).

primary    : one-sample t test on the G cluster means, t(G-1) reference (Reviewer 1 #3)
sensitivity: exact sign-flip (randomization) test over all 2^G sign vectors
             (smallest attainable two-sided p = 2 / 2^G, i.e. 1/64 for G = 7);
             wild cluster bootstrap-t with Webb six-point weights, null imposed,
             all 6^G weight vectors enumerated when G <= 7 (Webb 2023, Can J Econ 56:839-858;
             MacKinnon, Nielsen & Webb 2023, J Econom 232:272-299).
effect size: d_z = mean / SD of the cluster means (consistency across models, not
             magnitude relative to item-level variability); also the raw difference on
             the 1-7 scale (= 6 x a).
"""
import itertools
import numpy as np
from scipy import stats

WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def _clean(v):
    v = np.asarray(v, dtype=float)
    return v[~np.isnan(v)]


def t_summary(v):
    v = _clean(v)
    G = len(v)
    out = dict(G=G, mean=np.nan, sd=np.nan, se=np.nan, ci_lo=np.nan, ci_hi=np.nan,
               t=np.nan, df=G - 1, p_t=np.nan, d_z=np.nan, k_same_sign=np.nan, scale_points=np.nan)
    if G == 0:
        return out
    m = float(v.mean())
    out.update(mean=m, scale_points=6.0 * m, k_same_sign=int((np.sign(v) == np.sign(m)).sum()))
    if G < 2:
        return out
    sd = float(v.std(ddof=1))
    se = sd / np.sqrt(G)
    h = stats.t.ppf(0.975, G - 1) * se
    out.update(sd=sd, se=se, ci_lo=m - h, ci_hi=m + h)
    if se > 0:
        t = m / se
        out.update(t=t, p_t=float(2 * stats.t.sf(abs(t), G - 1)), d_z=m / sd)
    return out


def signflip_p(v, max_exact=16, n_mc=200000, seed=1):
    v = _clean(v)
    G = len(v)
    if G < 2:
        return np.nan
    obs = abs(v.mean())
    if G <= max_exact:
        S = np.array(list(itertools.product([1.0, -1.0], repeat=G)))
    else:
        S = np.random.default_rng(seed).choice([1.0, -1.0], size=(n_mc, G))
    return float((np.abs((S * v).mean(axis=1)) >= obs - 1e-12).mean())


def wild_webb_p(v, max_exact=7, n_mc=200000, seed=1):
    """Restricted (null-imposed) wild cluster bootstrap-t for H0: mean = 0."""
    v = _clean(v)
    G = len(v)
    if G < 3:
        return np.nan
    se = v.std(ddof=1) / np.sqrt(G)
    if se == 0:
        return np.nan
    t_obs = abs(v.mean() / se)
    if G <= max_exact:
        W = np.array(list(itertools.product(WEBB, repeat=G)))
    else:
        W = np.random.default_rng(seed).choice(WEBB, size=(n_mc, G))
    Y = W * v
    se_b = Y.std(axis=1, ddof=1) / np.sqrt(G)
    t_b = np.abs(Y.mean(axis=1) / np.where(se_b > 0, se_b, np.nan))
    return float(np.nanmean(t_b >= t_obs - 1e-12))


def bh(p):
    """Benjamini-Hochberg adjusted p-values (NaN-safe)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    n = len(q)
    if n == 0:
        return out
    order = np.argsort(q)
    ranked = q[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def full_summary(v):
    s = t_summary(v)
    s["p_signflip"] = signflip_p(v)
    s["p_wild_webb"] = wild_webb_p(v)
    return s


def fmt_p(p):
    """PLOS style: exact to 3 decimals when >= 0.001, otherwise '< 0.001'."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NA"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"
