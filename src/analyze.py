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
  S_family_level_as_submitted.csv  five-family grouping of the submitted version (S1 File)
  S_parser_corrected_*.csv       corrected parser, decision rule 2 of the validation protocol
  M1..M9_*.csv                   missingness decomposition (R1 #4, R2 #2); M9 by dimension
  S_accounting_by_cell.csv       expected / actual / valid by format x language x referent x model (R2 #5)
  H5_*.csv                       exploratory hedging / refusal (R2 #7)
  X_convergence.csv              exploratory rating vs free-text agreement
  S_consistency_*.csv            agreement across repeats, templates and languages (R2 #4)
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
    # lineage from config; the stored column keeps the submitted grouping (see src/measure.py)
    rr["family"] = rr["model"].map(config.family)
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
    elif mode == "clip_out_of_range":
        # answers outside 1-7 (e.g. "0" for an AI system) are clipped to the nearest scale end
        oor = d["category"] == "out_of_range"
        d.loc[oor, "v"] = d.loc[oor, "strict_raw_number"].clip(1.0, 7.0)
    elif mode in ("bound_human", "bound_machine"):
        ai_fill, hu_fill = (1.0, 7.0) if mode == "bound_human" else (7.0, 1.0)
        d.loc[d["v"].isna() & (d["agent"] == "ai"), "v"] = ai_fill
        d.loc[d["v"].isna() & (d["agent"] == "human"), "v"] = hu_fill
    return make_cells(d, col="v")


# ------------------------------------------------------------------- corrected parser
def corrected_parser(rr, rd):
    """Decision rule 2 of the registered parser-validation protocol: the misreading patterns found
    in coding are corrected (src/rating_parse_corrected.py), the analysis is repeated and both
    versions are reported. The strict parser stays primary; its accuracy is the one estimated."""
    from src import rating_parse_corrected as RC
    th = config.EXP["measurement"].get("truncation_heuristic", {}) or {}
    rc = RC.apply(rr, th.get("en_min_words", 6), th.get("zh_min_chars", 12))
    ok = rc["corrected_category"].isin(VALID_CATEGORIES)
    cells_c = make_cells(rc, col="corrected_value", valid=ok)
    md_c = model_dim(cells_c)

    ch = rc[rc["corrected_rule"] != ""]
    ch[["model", "language", "agent", "dimension", "template", "repeat", "prompt_id", "corrected_rule",
        "strict_value", "category", "corrected_value", "corrected_category", "raw_response"]] \
        .sort_values(["corrected_rule", "model", "language", "dimension"]) \
        .to_csv(rd / "S_parser_corrected_changes.csv", index=False)
    rules = {"R1": "range read as its lower end", "R2": "unfinished range", "R3": "scale listing",
             "R4": "anchor definition", "R5": "conflicting second answer"}
    summ = ch.groupby(["corrected_rule", "model"]).size().rename("n").reset_index()
    summ.insert(1, "pattern", summ["corrected_rule"].map(rules))
    summ.to_csv(rd / "S_parser_corrected_patterns.csv", index=False)

    out = pd.DataFrame({"responses": rc.groupby("model").size(),
                        "valid_strict": rc.groupby("model")["valid"].sum(),
                        "valid_corrected": ok.groupby(rc["model"]).sum()})
    out.loc["ALL"] = out.sum()
    out["pct_invalid_strict"] = 100 * (1 - out["valid_strict"] / out["responses"])
    out["pct_invalid_corrected"] = 100 * (1 - out["valid_corrected"] / out["responses"])
    out["changed"] = ch.groupby("model").size().reindex(out.index).fillna(0).astype(int)
    out.loc["ALL", "changed"] = len(ch)
    out["complete_cells_strict"] = make_cells(rr, col="strict_value", valid=rr["valid"]).groupby("model").size() \
        .reindex(out.index)
    out["complete_cells_corrected"] = cells_c.groupby("model").size().reindex(out.index)
    out.loc["ALL", ["complete_cells_strict", "complete_cells_corrected"]] = \
        [out["complete_cells_strict"].drop("ALL").sum(), out["complete_cells_corrected"].drop("ALL").sum()]
    out.reset_index(names="model").to_csv(rd / "S_parser_corrected_outcomes.csv", index=False)

    t = h2_table(md_c, label="corrected parser (validation rule 2)")
    pm = h1_per_model(md_c)
    t = pd.concat([t, pd.DataFrame([dict(spec="corrected parser (validation rule 2)", dimension="H1_overall",
                                         **full_summary(pm["h1_model_mean"].values))])])
    t.to_csv(rd / "S_parser_corrected_H2.csv", index=False)
    return cells_c, rc


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
    missingness_by_dimension(rr, cats, rd)
    return m1, m3, fp


def missingness_by_dimension(rr, cats, rd):
    """Invalid rate of each dimension and its outcome composition (Reviewer 1 #4, Reviewer 2 #2).
    Model-level: each model's rate in the dimension, then mean, min and max over models (each
    model weighted equally); pooled counts alongside. Whether dimensions differ: Friedman test over
    the model x dimension rates (models as blocks), with Kendall's W."""
    rate = rr.groupby(["model", "dimension"])["invalid"].mean().unstack("dimension").reindex(columns=dim_order())
    share = pd.crosstab([rr["model"], rr["dimension"]], rr["category"], normalize="index") \
        .reindex(columns=cats, fill_value=0.0)
    rows = []
    for d in dim_order() + ["ALL"]:
        sub = rr if d == "ALL" else rr[rr["dimension"] == d]
        r_ = rr.groupby("model")["invalid"].mean() if d == "ALL" else rate[d]
        sh = (pd.crosstab(rr["model"], rr["category"], normalize="index").reindex(columns=cats, fill_value=0.0)
              if d == "ALL" else share.xs(d, level="dimension"))
        row = dict(dimension=d, responses=len(sub), invalid=int(sub["invalid"].sum()),
                   pct_invalid_pooled=100 * sub["invalid"].mean(),
                   pct_invalid_model_mean=100 * r_.mean(), pct_invalid_model_min=100 * r_.min(),
                   model_min=r_.idxmin().split("/")[-1], pct_invalid_model_max=100 * r_.max(),
                   model_max=r_.idxmax().split("/")[-1],
                   pct_invalid_ai_model_mean=100 * sub[sub["agent"] == "ai"].groupby("model")["invalid"].mean().mean(),
                   pct_invalid_human_model_mean=100 * sub[sub["agent"] == "human"].groupby("model")["invalid"].mean().mean())
        row.update({f"pct_{c}_model_mean": 100 * sh[c].mean() for c in cats})
        rows.append(row)
    pd.DataFrame(rows).to_csv(rd / "M9_invalid_by_dimension.csv", index=False)
    x = rate.dropna()
    chi2, p = stats.friedmanchisquare(*[x[c].values for c in x.columns])
    n_, k_ = x.shape
    pd.DataFrame([dict(test="Friedman, invalid rate across dimensions (models as blocks)", models=n_, dimensions=k_,
                       chi2=chi2, df=k_ - 1, p=p, kendalls_W=chi2 / (n_ * (k_ - 1)))]) \
        .to_csv(rd / "M9b_invalid_dimension_test.csv", index=False)
    (100 * rate).round(2).to_csv(rd / "M9c_invalid_rate_model_by_dimension.csv")


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
    accounting_by_cell(rr, fr, rd)
    return t, n_rp, n_fp


def accounting_by_cell(rr, fr, rd):
    """Expected, received and usable responses by format x language x referent x model (Reviewer 2 #5).
    Expected = prompts in the June grid x the model's repeats. Usable = a valid rating (strict parser)
    for the rating format; a non-empty response for the free-text formats, which are archived but not
    analysed in the revision. Retained models only (excluded models: excluded_models_summary.csv)."""
    grid_p = config.path("prompts")
    if not os.path.exists(grid_p):
        return None
    g = pd.read_json(grid_p, lines=True)
    design = g.groupby(["format", "language", "agent"]).size()
    reps = rr.groupby("model")["repeat"].max() + 1
    rows = []
    for m in sorted(rr["model"].unique()):
        for (fmt, lang, agent), n in design.items():
            if fmt == "rating":
                sub = rr[(rr["model"] == m) & (rr["language"] == lang) & (rr["agent"] == agent)]
                usable = int(sub["valid"].sum())
            else:
                sub = fr[(fr["model"] == m) & (fr["format"] == fmt) & (fr["language"] == lang)] if not fr.empty else fr
                usable = int((sub["empty"] == 0).sum()) if len(sub) else 0
            rows.append(dict(model=m, format=fmt, language=lang, referent=agent, prompts=int(n), repeats=int(reps[m]),
                             expected=int(n * reps[m]), received=len(sub), usable=usable,
                             usable_means="valid rating" if fmt == "rating" else "non-empty (not analysed)"))
    t = pd.DataFrame(rows)
    t["not_received"] = t["expected"] - t["received"]
    t["pct_usable_of_expected"] = (100 * t["usable"] / t["expected"]).round(1)
    tot = t.groupby(["format", "language", "referent", "usable_means"], sort=False)[
        ["prompts", "expected", "received", "usable", "not_received"]].sum().reset_index()
    tot["model"], tot["repeats"] = "ALL", np.nan
    tot["prompts"] = tot["prompts"] // rr["model"].nunique()
    tot["pct_usable_of_expected"] = (100 * tot["usable"] / tot["expected"]).round(1)
    t = pd.concat([t, tot[t.columns]], ignore_index=True)
    t.to_csv(rd / "S_accounting_by_cell.csv", index=False)
    return t


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


# ------------------------------------------------------------------------ consistency (Reviewer 2 #4)
def _oneway_icc(groups):
    """One-way random-effects ICC from lists of values (unbalanced): ICC(1) for one response and
    ICC(1,k0) for the mean of k0 responses (k0 = the ANOVA 'average' group size)."""
    groups = [np.asarray(g, float) for g in groups if len(g) >= 2]
    if len(groups) < 3:
        return dict(icc1=np.nan, icc1_k=np.nan, k0=np.nan, sd_within=np.nan)
    N, g = sum(len(x) for x in groups), len(groups)
    grand = np.concatenate(groups).mean()
    ssb = sum(len(x) * (x.mean() - grand) ** 2 for x in groups)
    ssw = sum(((x - x.mean()) ** 2).sum() for x in groups)
    msb, msw = ssb / (g - 1), ssw / (N - g)
    k0 = (N - sum(len(x) ** 2 for x in groups) / N) / (g - 1)
    den1 = msb + (k0 - 1) * msw
    return dict(icc1=(msb - msw) / den1 if den1 > 0 else np.nan,
                icc1_k=(msb - msw) / msb if msb > 0 else np.nan, k0=k0, sd_within=float(np.sqrt(msw)))


def _pair_stats(x, y):
    j = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(j) < 3:
        return dict(n_pairs=len(j), r=np.nan, sign_agreement_pct=np.nan, mean_abs_diff=np.nan)
    return dict(n_pairs=len(j), r=float(np.corrcoef(j["x"], j["y"])[0, 1]),
                sign_agreement_pct=100 * float((np.sign(j["x"]) == np.sign(j["y"])).mean()),
                mean_abs_diff=float((j["x"] - j["y"]).abs().mean()))


def consistency(rr, cells, rd):
    """Consistency of the June rating data across repeats, paraphrases (templates) and languages.
    Writes S_consistency_repeats.csv (per model: agreement of the repeated answers to each prompt),
    S_consistency_split_half.csv (asymmetries from even vs odd repeats) and S_consistency_levels.csv
    (model x dimension asymmetry: repeat halves vs template pairs vs languages)."""
    k = ["model", "dimension", "language", "template", "agent"]
    rows = []
    for m, g in rr.groupby("model"):
        per = g.groupby(k)
        vals = [s["strict_value"][s["valid"]].values for _, s in per]
        vv = [v for v in vals if len(v) >= 2]
        valid_same = per["valid"].agg(lambda s: s.all() or (~s).all())
        icc = _oneway_icc(vv)
        rows.append(dict(model=m, repeats=int(g["repeat"].nunique()), prompts=len(vals),
                         prompts_2plus_valid=len(vv),
                         pct_identical=100 * np.mean([len(set(v)) == 1 for v in vv]) if vv else np.nan,
                         pct_range_le_1=100 * np.mean([v.max() - v.min() <= 1 for v in vv]) if vv else np.nan,
                         pct_modal=100 * np.mean([pd.Series(v).value_counts().iloc[0] / len(v) for v in vv]) if vv else np.nan,
                         pct_validity_same_all_repeats=100 * float(valid_same.mean()), **icc))
    rep = pd.DataFrame(rows)
    med = rep.drop(columns=["model"]).median(numeric_only=True)
    rep = pd.concat([rep, pd.DataFrame([dict(model="median over models", **med.to_dict())])], ignore_index=True)
    rep.to_csv(rd / "S_consistency_repeats.csv", index=False)

    # split halves of the repeats (even vs odd repeat index); model x dimension asymmetry per half
    halves = {}
    for lab, par in (("even_repeats", 0), ("odd_repeats", 1)):
        sub = rr[rr["repeat"] % 2 == par]
        halves[lab] = model_dim(make_cells(sub, col="strict_value", valid=sub["valid"])).set_index(["model", "dimension"])["a"]
    sh = []
    for d in dim_order():
        row = dict(dimension=d)
        for lab, s in halves.items():
            t = t_summary(s.xs(d, level="dimension").values)
            row.update({f"{lab}_mean": t["mean"], f"{lab}_ci_lo": t["ci_lo"], f"{lab}_ci_hi": t["ci_hi"],
                        f"{lab}_p_t": t["p_t"], f"{lab}_k": t["k_same_sign"], f"{lab}_G": t["G"]})
        sh.append(row)
    pd.DataFrame(sh).to_csv(rd / "S_consistency_split_half.csv", index=False)

    # model x dimension asymmetry: repeat halves, template pairs, languages
    lv = []
    s = _pair_stats(halves["even_repeats"], halves["odd_repeats"])
    s["spearman_brown"] = 2 * s["r"] / (1 + s["r"]) if pd.notna(s["r"]) else np.nan
    lv.append(dict(level="repeats: even vs odd half", **s))
    by_t = {t: model_dim(cells[cells["template"] == t]).set_index(["model", "dimension"])["a"]
            for t in sorted(cells["template"].unique())}
    tp = [dict(level=f"templates: {t1} vs {t2}", **_pair_stats(by_t[t1], by_t[t2]))
          for i, t1 in enumerate(by_t) for t2 in list(by_t)[i + 1:]]
    lv += tp
    if tp:
        tpd = pd.DataFrame(tp)
        lv.append(dict(level="templates: mean of the pairs", n_pairs=tpd["n_pairs"].mean(), r=tpd["r"].mean(),
                       sign_agreement_pct=tpd["sign_agreement_pct"].mean(), mean_abs_diff=tpd["mean_abs_diff"].mean()))
    by_l = {l: model_dim(cells[cells["language"] == l]).set_index(["model", "dimension"])["a"] for l in ("en", "zh")}
    lv.append(dict(level="languages: en vs zh", **_pair_stats(by_l["en"], by_l["zh"])))
    lvt = pd.DataFrame(lv)
    lvt.to_csv(rd / "S_consistency_levels.csv", index=False)
    return rep, lvt


# ------------------------------------------------------------------------- manifest
def manifest(rd, extra):
    def sha(p):
        # CRLF normalized to LF: the form git stores and GitHub/Zenodo serve, so the value is the same
        # on Windows (text files checked out with CRLF) and elsewhere, and `sha256sum` on a
        # downloaded copy reproduces it. All hashed inputs here are text (.jsonl).
        return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()

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
               input_sha256_note="SHA-256 with CRLF normalized to LF (the form stored in git, GitHub and Zenodo)",
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
    def family_level(md_, label):
        fam = md_.groupby(["family", "dimension"])["a"].mean().reset_index()
        ft = h2_table(fam.rename(columns={"family": "model"}).assign(family="-"), label=label)
        return pd.concat([ft, pd.DataFrame([dict(spec=label, dimension="H1_overall",
                                                 **full_summary(fam.groupby("family")["a"].mean().values))])])
    family_level(md, "family_level").to_csv(rd / "S_family_level.csv", index=False)
    # five-family grouping of the submitted version (Nemotron-Mini with Llama), for S1 File
    family_level(md.assign(family=md["model"].map(config.family_as_submitted)), "family_level_as_submitted") \
        .to_csv(rd / "S_family_level_as_submitted.csv", index=False)

    # ---- leave-one-out ----
    pd.DataFrame([r for m in sorted(md["model"].unique()) for r in quick(md[md["model"] != m], f"without {m}")]) \
        .to_csv(rd / "S_loo_model.csv", index=False)
    pd.DataFrame([r for f_ in sorted(md["family"].unique()) for r in quick(md[md["family"] != f_], f"without family {f_}")]) \
        .to_csv(rd / "S_loo_family.csv", index=False)

    # ---- corrected parser (decision rule 2 of the validation protocol; both versions reported) ----
    cells_corr, rc = corrected_parser(rr, rd)

    # ---- sensitivity grid ----
    sens = quick(md, "primary")
    sens += quick(model_dim(cells_legacy), "legacy parser (as submitted)")
    sens += quick(model_dim(cells_corr), "corrected parser (validation rule 2)")
    for lang in ("en", "zh"):
        sens += quick(model_dim(cells[cells["language"] == lang]), f"language {lang}")
    for tp in sorted(cells["template"].unique()):
        sens += quick(model_dim(cells[cells["template"] == tp]), f"template {tp}")
    hm = config.EXP.get("analysis", {}).get("high_missingness_models", []) or []
    sens += quick(model_dim(cells[~cells["model"].isin(hm)]), "excluding high-missingness models")
    sens += quick(model_dim(make_cells(rr, col="strict_value", valid=rr["valid"], min_frac=0.5)),
                  "cells with >= 50% valid on both sides")
    for mode, lab in (("clip_out_of_range", "out-of-range answers clipped to 1-7"),
                      ("neutral", "impute invalid: referent-neutral mean"), ("midpoint", "impute invalid: 4"),
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
    cons_rep, cons_lv = consistency(rr, cells, rd)

    # ---- missingness, accounting, H5, exploratory ----
    m1, m3, fp = missingness(rr, rd)
    acc, n_rp, n_fp = accounting(rr, fr, cells, cells_legacy, rd)
    t1 = m1.drop(index="ALL").reset_index()[["model", "total", "invalid", "pct_invalid", "empty", "refusal", "hedge",
                                             "truncated", "out_of_range", "multiple", "malformed"]]
    t1.insert(1, "family", t1["model"].map(config.family))
    t1.to_csv(rd / "T1_models.csv", index=False)
    conv = None
    if config.EXP.get("analysis", {}).get("report_freetext", False):
        h5(rr, fr, rd)
        conv = convergence(cells, fr, rd)
    else:
        for stale in ("H5_by_model.csv", "H5_by_stage.csv", "H5_freetext_by_model_format.csv", "X_convergence.csv"):
            (rd / stale).unlink(missing_ok=True)

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
             f"- Corrected parser (validation decision rule 2; S_parser_corrected_*.csv): "
             f"{int((rc['corrected_rule'] != '').sum())} responses re-read; invalid "
             f"{int((~rc['corrected_category'].isin(VALID_CATEGORIES)).sum())}; complete cells {len(cells_corr)}; "
             f"H1 {h1_per_model(model_dim(cells_corr))['h1_model_mean'].mean():+.3f}",
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
                  f"[{g0['ci_lo_pp']:+.1f}, {g0['ci_hi_pp']:+.1f}], p = {fmt_p(g0['p_t'])}, {int(g0['k_same_sign'])}/{int(g0['G'])} same sign"]
    m9 = pd.read_csv(rd / "M9_invalid_by_dimension.csv").set_index("dimension").drop(index="ALL")
    m9b = pd.read_csv(rd / "M9b_invalid_dimension_test.csv").iloc[0]
    lines += [f"- Invalid rate by dimension (mean over models, M9): {m9['pct_invalid_model_mean'].min():.1f}% "
              f"({m9['pct_invalid_model_mean'].idxmin()}) to {m9['pct_invalid_model_mean'].max():.1f}% "
              f"({m9['pct_invalid_model_mean'].idxmax()}); Friedman chi2({int(m9b['df'])}) = {m9b['chi2']:.2f}, "
              f"p = {fmt_p(m9b['p'])}, Kendall's W = {m9b['kendalls_W']:.2f}",
              "", "## Sensitivity (mean, * = unadjusted t p < 0.05, k/G = models sharing the sign)", "", _md_table(sw)]
    mr = cons_rep[cons_rep["model"] == "median over models"].iloc[0]
    lvi = cons_lv.set_index("level")
    lines += ["", "## Consistency (S_consistency_*.csv)", "",
              f"- Repeats of the same prompt (median over models): all valid answers identical in {mr['pct_identical']:.0f}% "
              f"of prompts, within 1 point in {mr['pct_range_le_1']:.0f}%; ICC(1) = {mr['icc1']:.2f} for one answer, "
              f"{mr['icc1_k']:.2f} for the mean of the repeats",
              f"- Model x dimension asymmetry, even vs odd repeats: r = {lvi.loc['repeats: even vs odd half', 'r']:.2f}; "
              f"between templates (mean of the pairs): r = {lvi.loc['templates: mean of the pairs', 'r']:.2f}; "
              f"English vs Chinese: r = {lvi.loc['languages: en vs zh', 'r']:.2f}"]
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
