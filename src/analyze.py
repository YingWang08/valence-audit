"""Analysis (revision R1). Reads the per-response tables written by src/measure.py and
writes every table used in the manuscript and S1 File to <data root>/results/.

Unit of analysis = model, each model weighted equally (Reviewer 3). For each dimension,
the model-level value is the mean of that model's cell asymmetries; H1 is the mean over
models of each model's dimension-balanced mean. Primary inference: t(G-1) with
Benjamini-Hochberg correction over the eight dimensions. Sensitivity: exact sign-flip,
Webb wild cluster bootstrap-t, family level, leave-one-model/family-out, language,
template, parser, missing-data imputations and worst-case bounds.

Main outputs (file prefix -> manuscript use)
  T1_models.csv                  Table 1 (per-model outcome rates)
  T2_accounting.csv              design and observation accounting (R1 #9, R2 #5)
  T3_H2_model_level.csv          Table 3 (primary)
  H1_overall.csv, H1_per_model.csv
  S_family_level.csv, S_loo_model.csv, S_loo_family.csv, S_sensitivity_long.csv,
  S_sensitivity_wide.csv, S_model_by_dimension.csv, S_language_by_dimension.csv
  M1..M8_*.csv                   missingness decomposition (R1 #4, R2 #2)
  H5_*.csv                       exploratory hedging / refusal (R2 #7)
  X_convergence.csv              exploratory rating vs free-text agreement
  C_submitted_vs_revised.csv     continuity: submitted numbers reproduced with the legacy parser
  results_summary.md, run_manifest.json
"""
import os
import json
import hashlib
import platform
import subprocess
import datetime as dt
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from src import config
from src.stats_utils import full_summary, t_summary, signflip_p, bh, fmt_p
from src.rating_parse import VALID_CATEGORIES, INVALID_CATEGORIES

warnings.filterwarnings("ignore")
KEY = ["model", "family", "alignment_stage", "dimension", "language", "template"]


# ----------------------------------------------------------------------------- helpers
def dim_order():
    order = config.EXP.get("analysis", {}).get("dimension_order")
    return order or list(config.DIMENSIONS)


def load():
    rr = pd.read_csv(config.path("rating_responses"), low_memory=False)
    rr = rr[~rr["model"].isin(config.excluded_models())].copy()
    rr["valid"] = rr["category"].isin(VALID_CATEGORIES)
    fr_path = config.path("freetext_responses")
    fr = pd.read_csv(fr_path, low_memory=False) if os.path.exists(fr_path) else pd.DataFrame()
    if not fr.empty:
        fr = fr[~fr["model"].isin(config.excluded_models())].copy()
    return rr, fr


def value_column():
    return "legacy_value" if config.EXP["measurement"].get("rating_parser", "strict") == "legacy" else "strict_value"


def make_cells(rr, col="strict_value", valid=None, min_frac=None):
    """Cell asymmetry a = (mean AI - mean human) / 6 from response rows with a value in `col`."""
    d = rr[rr["agent"].isin(["ai", "human"])]
    ok = d[col].notna() if valid is None else valid.loc[d.index]
    d = d[ok]
    g = d.groupby(KEY + ["agent"])[col].agg(["mean", "size"]).unstack("agent")
    g = g.dropna(subset=[("mean", "ai"), ("mean", "human")])
    out = pd.DataFrame({"mean_ai": g[("mean", "ai")], "mean_human": g[("mean", "human")],
                        "n_ai": g[("size", "ai")], "n_human": g[("size", "human")]}).reset_index()
    out["a"] = (out["mean_ai"] - out["mean_human"]) / 6.0
    if min_frac is not None and len(out):
        reps = rr.groupby("model")["repeat"].max() + 1
        need = out["model"].map(reps) * min_frac
        out = out[(out["n_ai"] >= need) & (out["n_human"] >= need)]
    return out


def model_dim(cells):
    return cells.groupby(["model", "family", "dimension"])["a"].mean().reset_index()


def h2_table(md, label="primary"):
    rows = []
    for d in dim_order():
        v = md.loc[md["dimension"] == d, "a"].values
        s = full_summary(v)
        exp = config.expected_sign(d)
        rows.append(dict(spec=label, dimension=d, expected_sign=exp, **s,
                         matches_prediction=bool(exp != 0 and np.sign(s["mean"]) == exp)))
    t = pd.DataFrame(rows)
    t["p_t_BH"] = bh(t["p_t"].values)
    t["p_signflip_BH"] = bh(t["p_signflip"].values)
    t["p_wild_webb_BH"] = bh(t["p_wild_webb"].values)
    return t


def h1_per_model(md):
    return md.groupby(["model", "family"])["a"].mean().reset_index().rename(columns={"a": "h1_model_mean"})


def quick(md, label):
    """Light version for sensitivity grids: mean, 95% CI, p_t, k/G per dimension and H1."""
    rows = []
    for d in dim_order():
        s = t_summary(md.loc[md["dimension"] == d, "a"].values)
        rows.append(dict(spec=label, dimension=d, G=s["G"], mean=s["mean"], ci_lo=s["ci_lo"],
                         ci_hi=s["ci_hi"], p_t=s["p_t"], k_same_sign=s["k_same_sign"]))
    h1 = t_summary(h1_per_model(md)["h1_model_mean"].values) if len(md) else t_summary([])
    rows.append(dict(spec=label, dimension="H1_overall", G=h1["G"], mean=h1["mean"], ci_lo=h1["ci_lo"],
                     ci_hi=h1["ci_hi"], p_t=h1["p_t"], k_same_sign=h1["k_same_sign"]))
    return rows


# ------------------------------------------------------------------ imputations and bounds
def imputed_cells(rr, mode):
    d = rr[rr["agent"].isin(["ai", "human"])].copy()
    d["v"] = d["strict_value"].where(d["valid"])
    if mode == "neutral":
        # referent-neutral: model x dimension x language mean of valid ratings pooled over both referents
        d["v"] = d["v"].fillna(d.groupby(["model", "dimension", "language"])["v"].transform("mean"))
    elif mode == "midpoint":
        d["v"] = d["v"].fillna(4.0)
    elif mode in ("bound_human", "bound_machine"):
        ai_fill, hu_fill = (1.0, 7.0) if mode == "bound_human" else (7.0, 1.0)
        d.loc[d["v"].isna() & (d["agent"] == "ai"), "v"] = ai_fill
        d.loc[d["v"].isna() & (d["agent"] == "human"), "v"] = hu_fill
    return make_cells(d, col="v")


# ----------------------------------------------------------------------- missingness
def missingness(rr, rd):
    rr = rr.copy()
    rr["invalid"] = (~rr["valid"]).astype(int)
    cats = list(VALID_CATEGORIES) + list(INVALID_CATEGORIES)

    m1 = pd.crosstab(rr["model"], rr["category"]).reindex(columns=cats, fill_value=0)
    m1["total"] = m1[cats].sum(axis=1)
    m1["invalid"] = m1[list(INVALID_CATEGORIES)].sum(axis=1)
    m1.loc["ALL"] = m1.sum(numeric_only=True)
    m1["pct_invalid"] = 100 * m1["invalid"] / m1["total"]
    m1["pct_invalid_excl_empty"] = 100 * (m1["invalid"] - m1["empty"]) / m1["total"]
    m1.to_csv(rd / "M1_outcomes_by_model.csv")

    rows = []
    for (m, lang), g in rr.groupby(["model", "language"]):
        a, h = g[g["agent"] == "ai"], g[g["agent"] == "human"]
        tab = [[int(a["invalid"].sum()), int(len(a) - a["invalid"].sum())],
               [int(h["invalid"].sum()), int(len(h) - h["invalid"].sum())]]
        p = stats.fisher_exact(tab)[1] if len(a) and len(h) else np.nan
        rows.append(dict(model=m, language=lang, n_ai=len(a), invalid_ai=tab[0][0],
                         pct_invalid_ai=100 * a["invalid"].mean() if len(a) else np.nan,
                         n_human=len(h), invalid_human=tab[1][0],
                         pct_invalid_human=100 * h["invalid"].mean() if len(h) else np.nan, fisher_p=p))
    pd.DataFrame(rows).to_csv(rd / "M2_invalid_by_model_referent_language.csv", index=False)

    gap = rr.groupby(["model", "dimension", "agent"])["invalid"].mean().unstack("agent")
    gap["gap"] = gap["ai"] - gap["human"]
    gap = gap.reset_index()
    rows = []
    for d in ["ALL"] + dim_order():
        sub = gap if d == "ALL" else gap[gap["dimension"] == d]
        v = sub.groupby("model")["gap"].mean().values
        s = t_summary(v)
        rows.append(dict(dimension=d, G=s["G"], mean_gap_pp=100 * s["mean"], ci_lo_pp=100 * s["ci_lo"],
                         ci_hi_pp=100 * s["ci_hi"], p_t=s["p_t"], p_signflip=signflip_p(v),
                         k_same_sign=s["k_same_sign"],
                         mean_invalid_ai_pct=100 * sub.groupby("model")["ai"].mean().mean(),
                         mean_invalid_human_pct=100 * sub.groupby("model")["human"].mean().mean()))
    m3 = pd.DataFrame(rows)
    m3["p_t_BH_dims"] = np.nan
    m3.loc[m3["dimension"] != "ALL", "p_t_BH_dims"] = bh(m3.loc[m3["dimension"] != "ALL", "p_t"].values)
    m3.to_csv(rd / "M3_referent_gap_by_dimension.csv", index=False)

    rr.groupby(["model", "dimension", "agent", "language", "template", "category"]).size() \
        .rename("n").reset_index().to_csv(rd / "M4_crosstab_model_dimension_referent_language_template.csv", index=False)
    rr.groupby(["model", "dimension", "agent", "language"])["invalid"].agg(["size", "sum", "mean"]) \
        .rename(columns={"size": "n", "sum": "invalid", "mean": "rate"}).reset_index() \
        .to_csv(rd / "M5_invalid_rate_model_dimension_referent_language.csv", index=False)

    fp = rr[rr["legacy_value"].notna() & ~rr["valid"]]
    mm = rr[rr["valid"] & rr["legacy_value"].notna() & (rr["legacy_value"] != rr["strict_value"])]
    grp = ["model", "language", "agent"]
    m6 = pd.DataFrame({"n_responses": rr.groupby(grp).size(),
                       "legacy_false_positive": fp.groupby(grp).size(),
                       "legacy_fp_mean_value": fp.groupby(grp)["legacy_value"].mean(),
                       "legacy_value_differs_on_valid": mm.groupby(grp).size()}) \
        .fillna({"legacy_false_positive": 0, "legacy_value_differs_on_valid": 0}).reset_index()
    m6.to_csv(rd / "M6_legacy_vs_strict_parser.csv", index=False)
    fp.groupby(["category", "language"])["legacy_value"].value_counts().rename("n").reset_index() \
        .to_csv(rd / "M6b_legacy_false_positive_values.csv", index=False)

    pd.crosstab([rr["model"], rr["category"]], rr["trunc_flag"]) \
        .rename(columns={0: "not_truncated", 1: "truncated_flag"}).to_csv(rd / "M7_truncation_by_category.csv")

    m8 = rr.groupby(["model", "template", "agent"])["invalid"].agg(["size", "sum", "mean"]).reset_index()
    m8.columns = ["model", "template", "agent", "n", "invalid", "rate"]
    m8.to_csv(rd / "M8_invalid_by_template_referent.csv", index=False)
    return m1, m3, fp


# ------------------------------------------------------------------------ accounting
def accounting(rr, fr, cells, cells_legacy, rd):
    grid_p = config.path("prompts")
    if os.path.exists(grid_p):
        g = pd.read_json(grid_p, lines=True)
        n_rp, n_fp = int((g["format"] == "rating").sum()), int((g["format"] != "rating").sum())
    else:
        n_rp = int(rr["prompt_id"].nunique())
        n_fp = int(fr["prompt_id"].nunique()) if not fr.empty else 0
    n_cells_design = n_rp // 2
    rows = []
    for m, g in rr.groupby("model"):
        reps = int(g["repeat"].max()) + 1
        n_complete = int((cells["model"] == m).sum())
        sides = g[g["valid"]].groupby(["dimension", "language", "template"])["agent"].nunique()
        n_one = int((sides == 1).sum())
        f = fr[fr["model"] == m] if not fr.empty else pd.DataFrame()
        rows.append(dict(model=m, family=g["family"].iloc[0], stage=g["alignment_stage"].iloc[0], repeats=reps,
                         rating_calls_designed=n_rp * reps, rating_responses=len(g),
                         rating_empty=int((g["category"] == "empty").sum()), rating_valid=int(g["valid"].sum()),
                         rating_invalid_nonempty=int((~g["valid"] & (g["category"] != "empty")).sum()),
                         pct_rating_invalid=round(100 * (~g["valid"]).mean(), 1),
                         cells_designed=n_cells_design, cells_complete=n_complete, cells_one_sided=n_one,
                         cells_no_valid=n_cells_design - n_complete - n_one,
                         cells_complete_legacy=int((cells_legacy["model"] == m).sum()),
                         freetext_calls_designed=n_fp * reps, freetext_responses=len(f),
                         freetext_empty=int(f["empty"].sum()) if len(f) else 0,
                         freetext_text_m1=int(f["text_m1"].notna().sum()) if len(f) else 0))
    t = pd.DataFrame(rows)
    tot = t.drop(columns=["repeats"]).select_dtypes("number").sum()
    tot["pct_rating_invalid"] = round(100 * (1 - tot["rating_valid"] / tot["rating_responses"]), 1)
    t = pd.concat([t, pd.DataFrame([{"model": "TOTAL", **tot.to_dict()}])], ignore_index=True)
    t.to_csv(rd / "T2_accounting.csv", index=False)
    return t, n_rp, n_fp


# ------------------------------------------------------------------------------ H5
def h5(rr, fr, rd):
    out = []
    if not fr.empty:
        for (m, st), g in fr.groupby(["model", "alignment_stage"]):
            out.append(dict(model=m, stage=st, format="free-text (all)", n=len(g),
                            hedge_k=int(g["hedge"].sum()), refuse_k=int(g["refuse"].sum())))
    for (m, st), g in rr.groupby(["model", "alignment_stage"]):
        out.append(dict(model=m, stage=st, format="rating", n=len(g),
                        hedge_k=int((g["category"] == "hedge").sum()), refuse_k=int((g["category"] == "refusal").sum())))
    t = pd.DataFrame(out)
    t["hedge_pct"] = 100 * t["hedge_k"] / t["n"]
    t["refuse_pct"] = 100 * t["refuse_k"] / t["n"]
    t.to_csv(rd / "H5_by_model.csv", index=False)
    st = t.groupby(["format", "stage"])[["n", "hedge_k", "refuse_k"]].sum().reset_index()
    st["hedge_pct"] = 100 * st["hedge_k"] / st["n"]
    st["refuse_pct"] = 100 * st["refuse_k"] / st["n"]
    st.to_csv(rd / "H5_by_stage.csv", index=False)
    if not fr.empty:
        fr.groupby(["model", "format"])[["hedge", "refuse"]].mean().mul(100).reset_index() \
            .to_csv(rd / "H5_freetext_by_model_format.csv", index=False)
    return t, st


# ------------------------------------------------------------------------ exploratory
def _icc(X):
    X = np.asarray(X, float)
    n, k = X.shape
    grand = X.mean()
    SSR = k * ((X.mean(1) - grand) ** 2).sum()
    SSC = n * ((X.mean(0) - grand) ** 2).sum()
    SSE = ((X - grand) ** 2).sum() - SSR - SSC
    MSR, MSC, MSE = SSR / (n - 1), SSC / (k - 1), SSE / ((n - 1) * (k - 1))
    return dict(ICC2_1=(MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n),
                ICC2_k=(MSR - MSE) / (MSR + (MSC - MSE) / n))


def convergence(cells, fr, rd):
    if fr.empty or fr["text_m1"].notna().sum() == 0:
        return None
    a = cells.groupby(["model", "dimension", "language"])["a"].mean()
    b = fr.dropna(subset=["text_m1"]).groupby(["model", "dimension", "language"])["text_m1"].mean()
    j = pd.concat([a.rename("rating"), b.rename("text_m1")], axis=1).dropna()
    if len(j) < 3 or j["text_m1"].std() == 0 or j["rating"].std() == 0:
        return None
    z = (j - j.mean()) / j.std(ddof=0)
    res = dict(n_cells=len(j), pearson=stats.pearsonr(j["rating"], j["text_m1"])[0],
               spearman=stats.spearmanr(j["rating"], j["text_m1"])[0], **_icc(z.values))
    pd.DataFrame([res]).to_csv(rd / "X_convergence.csv", index=False)
    return res


# ------------------------------------------------------------------------- manifest
def manifest(rd, extra):
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=config.ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    vers = {}
    for mod in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:
            pass
    # Outputs under data/ are rewritten by every run, so "dirty" is judged on code and inputs only.
    code_changes = git("status", "--porcelain", "--", ".", ":(exclude)data", ":(exclude)data_mock")
    input_changes = git("status", "--porcelain", "--", "data/raw", "data/prompts", "data/raw_quarantine")
    man = dict(timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               git_commit=git("rev-parse", "HEAD"), git_describe=git("describe", "--tags", "--always"),
               code_uncommitted=None if code_changes is None else bool(code_changes),
               code_uncommitted_files=(code_changes or "").splitlines()[:30],
               inputs_uncommitted=None if input_changes is None else bool(input_changes),
               inputs_uncommitted_files=(input_changes or "").splitlines()[:30],
               python=platform.python_version(), packages=vers,
               rating_parser=config.EXP["measurement"].get("rating_parser"),
               excluded_models=config.excluded_models(),
               input_sha256={p.name: sha(p) for p in sorted(config.raw_dir().glob("*.jsonl"))}, **extra)
    with open(rd / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2, ensure_ascii=False, default=str)
    return man


def _md_table(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        out.append("| " + " | ".join([str(idx)] + [str(x) for x in row.values]) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------------------ run
def run():
    rd = config.results_dir()
    rr, fr = load()
    col = value_column()
    G = rr["model"].nunique()
    print(f"\n========== analysis: {G} models, {len(rr)} rating responses, parser = "
          f"{config.EXP['measurement'].get('rating_parser')} ==========")

    valid_mask = rr["valid"] if col == "strict_value" else rr["legacy_value"].notna()
    cells = make_cells(rr, col=col, valid=valid_mask)
    cells_legacy = make_cells(rr, col="legacy_value", valid=rr["legacy_value"].notna())
    cells.to_csv(rd / "cells_primary.csv", index=False)
    md = model_dim(cells)

    # ---- H2 and H1 (primary) ----
    t3 = h2_table(md)
    t3.to_csv(rd / "T3_H2_model_level.csv", index=False)
    pm = h1_per_model(md).merge(cells.groupby("model")["a"].mean().rename("h1_model_mean_cell_weighted"), on="model")
    pm.to_csv(rd / "H1_per_model.csv", index=False)
    h1 = full_summary(pm["h1_model_mean"].values)
    h1_cw = full_summary(pm["h1_model_mean_cell_weighted"].values)
    pd.DataFrame([dict(version="dimension-balanced model means (primary)", **h1),
                  dict(version="cell-weighted within model", **h1_cw),
                  dict(version="pooled over cells (as submitted; no inference)", G=len(cells), mean=cells["a"].mean())]) \
        .to_csv(rd / "H1_overall.csv", index=False)
    md.pivot_table(index="dimension", columns="model", values="a").reindex(dim_order()).to_csv(rd / "S_model_by_dimension.csv")

    print(f"[H1] mean of model means = {h1['mean']:+.3f}  95% CI [{h1['ci_lo']:+.3f}, {h1['ci_hi']:+.3f}]  "
          f"t({h1['df']}) = {h1['t']:.2f}, p = {fmt_p(h1['p_t'])}; sign-flip p = {fmt_p(h1['p_signflip'])}; "
          f"{h1['k_same_sign']}/{h1['G']} same sign; per-model range {pm['h1_model_mean'].min():+.3f} to "
          f"{pm['h1_model_mean'].max():+.3f}")
    print("[H2] dimension      mean    95% CI             p_t(BH)  signflip   wild   k/G   d_z   predicted")
    for _, r in t3.iterrows():
        print(f"     {r['dimension']:13s} {r['mean']:+.3f}  [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  "
              f"{fmt_p(r['p_t_BH']):>7s}  {fmt_p(r['p_signflip']):>7s}  {fmt_p(r['p_wild_webb']):>6s}  "
              f"{int(r['k_same_sign'])}/{int(r['G'])}  {r['d_z']:+.2f}  {'match' if r['matches_prediction'] else 'no'}")

    # ---- family level (hierarchical sensitivity, Reviewer 1 #13) ----
    fam = md.groupby(["family", "dimension"])["a"].mean().reset_index()
    ft = h2_table(fam.rename(columns={"family": "model"}).assign(family="-"), label="family_level")
    ft = pd.concat([ft, pd.DataFrame([dict(spec="family_level", dimension="H1_overall",
                                           **full_summary(fam.groupby("family")["a"].mean().values))])])
    ft.to_csv(rd / "S_family_level.csv", index=False)

    # ---- leave-one-out ----
    pd.DataFrame([r for m in sorted(md["model"].unique()) for r in quick(md[md["model"] != m], f"without {m}")]) \
        .to_csv(rd / "S_loo_model.csv", index=False)
    pd.DataFrame([r for f_ in sorted(md["family"].unique()) for r in quick(md[md["family"] != f_], f"without family {f_}")]) \
        .to_csv(rd / "S_loo_family.csv", index=False)

    # ---- sensitivity grid ----
    sens = quick(md, "primary")
    sens += quick(model_dim(cells_legacy), "legacy parser (as submitted)")
    for lang in ("en", "zh"):
        sens += quick(model_dim(cells[cells["language"] == lang]), f"language {lang}")
    for tp in sorted(cells["template"].unique()):
        sens += quick(model_dim(cells[cells["template"] == tp]), f"template {tp}")
    hm = config.EXP.get("analysis", {}).get("high_missingness_models", []) or []
    sens += quick(model_dim(cells[~cells["model"].isin(hm)]), "excluding high-missingness models")
    sens += quick(model_dim(make_cells(rr, col="strict_value", valid=rr["valid"], min_frac=0.5)),
                  "cells with >= 50% valid on both sides")
    for mode, lab in (("neutral", "impute invalid: referent-neutral mean"), ("midpoint", "impute invalid: 4"),
                      ("bound_human", "bound: invalid AI=1, human=7"), ("bound_machine", "bound: invalid AI=7, human=1")):
        sens += quick(model_dim(imputed_cells(rr, mode)), lab)
    sl = pd.DataFrame(sens)
    sl.to_csv(rd / "S_sensitivity_long.csv", index=False)
    sl["cell"] = [f"{m:+.3f}{'*' if (pd.notna(p) and p < 0.05) else ''} ({int(k)}/{int(g)})" if pd.notna(m) and pd.notna(k) else "NA"
                  for m, p, k, g in zip(sl["mean"], sl["p_t"], sl["k_same_sign"], sl["G"])]
    sw = sl.pivot(index="dimension", columns="spec", values="cell").reindex(dim_order() + ["H1_overall"])[list(dict.fromkeys(sl["spec"]))]
    sw.to_csv(rd / "S_sensitivity_wide.csv")
    pd.concat([h2_table(model_dim(cells[cells["language"] == lang]), label=f"language {lang}") for lang in ("en", "zh")]) \
        .to_csv(rd / "S_language_by_dimension.csv", index=False)

    # ---- missingness, accounting, H5, exploratory ----
    m1, m3, fp = missingness(rr, rd)
    acc, n_rp, n_fp = accounting(rr, fr, cells, cells_legacy, rd)
    t1 = m1.drop(index="ALL").reset_index()[["model", "total", "invalid", "pct_invalid", "empty", "refusal", "hedge",
                                             "truncated", "out_of_range", "multiple", "malformed"]]
    t1.insert(1, "family", t1["model"].map(config.family))
    t1.to_csv(rd / "T1_models.csv", index=False)
    h5(rr, fr, rd)
    conv = convergence(cells, fr, rd)

    # ---- continuity with the submitted analysis ----
    lm = model_dim(cells_legacy)
    cont = [dict(dimension=d, pooled_cells_legacy_as_submitted=cells_legacy.loc[cells_legacy["dimension"] == d, "a"].mean(),
                 model_level_legacy=lm.loc[lm["dimension"] == d, "a"].mean(),
                 model_level_strict=md.loc[md["dimension"] == d, "a"].mean()) for d in dim_order()]
    cont.append(dict(dimension="H1_overall", pooled_cells_legacy_as_submitted=cells_legacy["a"].mean(),
                     model_level_legacy=h1_per_model(lm)["h1_model_mean"].mean(), model_level_strict=h1["mean"]))
    pd.DataFrame(cont).to_csv(rd / "C_submitted_vs_revised.csv", index=False)

    # ---- summary ----
    n, inv = len(rr), int((~rr["valid"]).sum())
    lines = ["# Results summary (auto-generated by src/analyze.py)", "",
             f"- Models: {G}; rating responses: {n}; invalid (strict parser): {inv} ({100 * inv / n:.1f}%); "
             f"complete cells: {len(cells)} of {G * (n_rp // 2)} designed",
             f"- Legacy (v1.0.0) parser: {len(cells_legacy)} complete cells; it assigned a number to {len(fp)} responses "
             f"that the strict parser classifies as non-ratings",
             f"- H1: {h1['mean']:+.3f} [{h1['ci_lo']:+.3f}, {h1['ci_hi']:+.3f}], t({h1['df']}) = {h1['t']:.2f}, "
             f"p = {fmt_p(h1['p_t'])}; sign-flip p = {fmt_p(h1['p_signflip'])}; {h1['k_same_sign']}/{G} models share the sign", "",
             "| Dimension | Mean a | 95% CI (t) | 1-7 points | p (t, BH) | p sign-flip | p wild (Webb) | k/G | d_z | Predicted |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t3.iterrows():
        lines.append(f"| {r['dimension']} | {r['mean']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['scale_points']:+.2f} | "
                     f"{fmt_p(r['p_t_BH'])} | {fmt_p(r['p_signflip'])} | {fmt_p(r['p_wild_webb'])} | "
                     f"{int(r['k_same_sign'])}/{int(r['G'])} | {r['d_z']:+.2f} | "
                     f"{'+' if r['expected_sign'] > 0 else '-'} ({'match' if r['matches_prediction'] else 'no match'}) |")
    g0 = m3.iloc[0]
    lines += ["", f"- Invalid-rate gap, AI minus human referent (model-level, percentage points): {g0['mean_gap_pp']:+.1f} "
                  f"[{g0['ci_lo_pp']:+.1f}, {g0['ci_hi_pp']:+.1f}], p = {fmt_p(g0['p_t'])}, {int(g0['k_same_sign'])}/{int(g0['G'])} same sign",
              "", "## Sensitivity (mean, * = unadjusted t p < 0.05, k/G = models sharing the sign)", "", _md_table(sw)]
    if conv:
        lines += ["", f"- Exploratory rating vs free-text: n = {conv['n_cells']}, r = {conv['pearson']:.2f}, ICC(2,1) = {conv['ICC2_1']:.3f}"]
    with open(rd / "results_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    manifest(rd, dict(n_models=G, n_rating_responses=n, n_invalid=inv, n_cells=len(cells),
                      n_cells_legacy=len(cells_legacy), n_rating_prompts=n_rp, n_freetext_prompts=n_fp))
    print(f"\n[analyze] {n} responses -> {len(cells)} complete cells; invalid {100 * inv / n:.1f}%")
    print(f"[analyze] tables written to {rd} (start with results_summary.md)")
    return t3


if __name__ == "__main__":
    run()
