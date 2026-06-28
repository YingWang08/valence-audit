"""分析阶段。读 items.parquet，跑 H1/H2/H3/H5 + 收敛效度（含 ICC 测量间信度）+ 稳健性，结果存 data/results/。

统计口径（与稿件一致）：
  - 分析单元 = 模型；H1/H2 用 cluster bootstrap（重抽样模型）给均值/95%CI/双侧 p；H2 再 FDR(BH) 校正；并报 Cohen's d。
  - H3：base/instruct 子集，关联（非因果）；缺 base 时干净跳过，不影响 H1/H2/H5。
  - H5：对冲/拒答率按对齐阶段。
  - 收敛效度：rating↔text_m1（及 m1↔m2）在 model×dim×lang cell 上的相关 + 测量间信度 ICC（construct validity）。
  - by_family：按架构家族聚合（模型非独立性的 Limitations 证据）。留一模型稳健性。
交叉随机效应（model+frame 同时）最终用 R 的 lme4；见 README / analyze_lme4.R。

★ 本文件可整文件替换旧 analyze.py。只依赖 numpy/pandas/scipy/statsmodels（requirements 里已有），无新依赖。
"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from src import config

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)
BOOT = 3000                                  # cluster bootstrap 次数


def _primary(df):
    """主指标优先用 rating；没有 rating 就退回 text_m1。"""
    if (df["measure"] == "rating").any():
        return df[df["measure"] == "rating"].copy(), "rating"
    return df[df["measure"] == "text_m1"].copy(), "text_m1"


def _safe_mixed(formula, data, group="model"):
    """模型组数<2 时 statsmodels 无法拟合，退回 OLS。返回 (params, pvalues)。（仅 H3 用）"""
    if data[group].nunique() >= 2 and len(data) > data[group].nunique():
        try:
            m = smf.mixedlm(formula, data, groups=data[group]).fit(method="lbfgs")
            return m.fe_params, m.pvalues
        except Exception:
            pass
    m = smf.ols(formula, data).fit()
    return m.params, m.pvalues


def _cohen_d(x):
    """单样本 Cohen's d（相对 0）：mean/sd。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 1e-12 else np.nan


def _boot_ci_p(d, by="model", value="asymmetry", B=BOOT):
    """Cluster bootstrap：重抽样 `by`（模型）这一聚类单元，返回 (mean, lo, hi, p_two_sided)。
    分析单元落在模型而非题项，从根上避免伪重复。模型数<2 时退回行级 percentile bootstrap。"""
    obs = float(d[value].mean()) if len(d) else np.nan
    groups = d[by].unique()
    if len(groups) >= 2:
        arrs = {g: d.loc[d[by] == g, value].values for g in groups}
        means = np.empty(B)
        for i in range(B):
            pick = rng.choice(groups, size=len(groups), replace=True)
            means[i] = np.concatenate([arrs[g] for g in pick]).mean()
    else:
        vals = d[value].values
        if len(vals) < 2:
            return obs, np.nan, np.nan, np.nan
        means = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(B)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())   # 双侧 bootstrap p
    p = min(1.0, max(p, 1.0 / B))                             # 下限地板，避免报 0
    return obs, float(lo), float(hi), float(p)


def _family(model_name):
    """由模型名解析架构家族（血缘/蒸馏优先），用于模型非独立性分析。"""
    s = str(model_name).lower()
    if "nemotron" in s:                                  return "llama"   # Nemotron 基于 Llama
    if "distill-qwen" in s or ("deepseek" in s and "qwen" in s): return "qwen"
    if "llama" in s:                                     return "llama"
    if "mixtral" in s or "mistral" in s:                 return "mistral"
    if "qwen" in s:                                      return "qwen"
    if "gemma" in s:                                     return "gemma"
    if "phi" in s:                                       return "phi"
    if "granite" in s:                                   return "granite"
    if "deepseek" in s:                                  return "deepseek"
    if "gpt-oss" in s or "gpt_oss" in s:                 return "openai-oss"
    if "glm" in s:                                       return "glm"
    if s.startswith("yi") or "yi-" in s:                 return "yi"
    if "mock" in s:                                      return "mock"
    return s.split("/")[0] if "/" in s else "other"


# ─────────────────── 测量间信度 ICC（construct validity）───────────────────
def _icc(matrix):
    """Shrout & Fleiss (1979) ICC，输入 n×k 矩阵（n=cell 行，k=测量列）。无外部依赖。
    数值已与 pingouin 交叉验证一致（ICC2_1≡ICC(A,1)，ICC3_1≡ICC(C,1) ...）。"""
    X = np.asarray(matrix, dtype=float)
    n, k = X.shape
    if n < 2 or k < 2:
        return None
    grand = X.mean()
    row_means = X.mean(axis=1)
    col_means = X.mean(axis=0)
    SST = float(((X - grand) ** 2).sum())
    SSR = float(k * ((row_means - grand) ** 2).sum())
    SSC = float(n * ((col_means - grand) ** 2).sum())
    SSE = SST - SSR - SSC
    SSW = SST - SSR
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else np.nan
    MSW = SSW / (n * (k - 1)) if n * (k - 1) > 0 else np.nan

    def _d(num, den):
        if den is None or np.isnan(den) or den == 0:
            return np.nan
        return float(num / den)

    return {"n_cells": n, "k_measures": k,
            "ICC1_1": _d(MSR - MSW, MSR + (k - 1) * MSW),
            "ICC2_1": _d(MSR - MSE, MSR + (k - 1) * MSE + k * (MSC - MSE) / n),
            "ICC3_1": _d(MSR - MSE, MSR + (k - 1) * MSE),
            "ICC1_k": _d(MSR - MSW, MSR),
            "ICC2_k": _d(MSR - MSE, MSR + (MSC - MSE) / n),
            "ICC3_k": _d(MSR - MSE, MSR)}


def convergence_icc(df, rd, cell_keys=("model", "dimension", "language"),
                    measures=("rating", "text_m1", "text_m2")):
    """测量间信度：每个测量在 cell 上求均值、按测量 z 标准化后，以测量为 rater、cell 为 subject 算 ICC。
    写 convergence_icc.csv 并打印；返回 headline ICC(2,1)。可用测量<2 个时返回 None。"""
    cols = {}
    for m in measures:
        s = df[df["measure"] == m]
        if s.empty:
            continue
        cm = s.groupby(list(cell_keys))["asymmetry"].mean()
        sd = cm.std(ddof=0)
        if sd and sd > 1e-9:
            cm = (cm - cm.mean()) / sd
        cols[m] = cm
    if len(cols) < 2:
        print("\n[ICC] 可用测量不足 2 个（默认 rating+text_m1；开 M2 后 3 个），跳过测量间信度。")
        return None
    mat = pd.concat(cols, axis=1).dropna()
    if len(mat) < 2:
        print("\n[ICC] 公共 cell 不足，跳过。")
        return None
    res = _icc(mat.values)
    res = {"measures": "+".join(mat.columns), **res}
    pd.DataFrame([res]).to_csv(rd / "convergence_icc.csv", index=False)
    print(f"\n[ICC] 测量间信度（{res['measures']}, n_cells={res['n_cells']}, k={res['k_measures']}）:")
    print(f"      ICC(2,1) 绝对一致·单测量 = {res['ICC2_1']:.3f}   ← 稿件 {{ICC_measures}} 用这个（最保守）")
    print(f"      ICC(2,k) 绝对一致·k测量均值 = {res['ICC2_k']:.3f}")
    print(f"      ICC(3,1) = {res['ICC3_1']:.3f}   ICC(3,k) = {res['ICC3_k']:.3f}")
    return res["ICC2_1"]


def run():
    rd = config.results_dir()
    df = pd.read_parquet(config.path("measured"))
    prim, prim_name = _primary(df)
    summary = []
    print(f"\n========== 分析（主指标 = {prim_name}, cluster bootstrap B={BOOT}）==========")

    # ---- H1 总体不对称：bootstrap 均值/CI/p + Cohen's d ----
    mean, lo, hi, p = _boot_ci_p(prim)
    d_h1 = _cohen_d(prim["asymmetry"].values)
    print(f"[H1] 总体不对称 = {mean:+.3f}  95%CI[{lo:+.3f}, {hi:+.3f}]  p={p:.2g}  d={d_h1:+.2f}  (>0 抬高机器)")
    summary.append(["H1_overall_asymmetry", mean, lo, hi, p, d_h1])

    # ---- H2 分维度：bootstrap CI + p → FDR(BH) + Cohen's d ----
    rows = []
    for dim in sorted(prim["dimension"].unique()):
        sub = prim[prim["dimension"] == dim]
        m, dlo, dhi, dp = _boot_ci_p(sub)
        dd = _cohen_d(sub["asymmetry"].values)
        exp = config.DIMENSIONS.get(dim, {}).get("expected_sign", "")
        rows.append([dim, m, dlo, dhi, dp, dd, exp])
    h2 = pd.DataFrame(rows, columns=["dimension", "asymmetry", "ci_lo", "ci_hi", "p_raw", "cohen_d", "expected"])
    rej, p_fdr, _, _ = multipletests(h2["p_raw"].fillna(1.0).values, method="fdr_bh")
    h2["p_fdr"], h2["sig"] = p_fdr, rej

    def _exp_sign(e):
        e = str(e).strip()
        return 1 if e.startswith("+") else (-1 if e.startswith("-") else 0)
    h2["matches_expected"] = [(_exp_sign(e) != 0 and np.sign(a) == _exp_sign(e))
                              for a, e in zip(h2["asymmetry"], h2["expected"])]
    h2 = h2.sort_values("asymmetry", ascending=False)[
        ["dimension", "asymmetry", "ci_lo", "ci_hi", "p_fdr", "sig", "cohen_d", "expected", "matches_expected"]]
    h2.to_csv(rd / "H2_by_dimension.csv", index=False)
    print("\n[H2] 分维度不对称（cluster bootstrap + FDR + Cohen's d）:")
    for _, r in h2.iterrows():
        ok = "✓" if r["matches_expected"] else " "
        print(f"   {r['dimension']:12s} {r['asymmetry']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
              f" p_FDR={r['p_fdr']:.2g} {'*' if r['sig'] else ' '}  d={r['cohen_d']:+.2f}  期望:{r['expected']} {ok}")

    # ---- H3 base vs instruct（关联，非因果）----
    stages = set(prim["alignment_stage"].unique())
    if {"base", "instruct"}.issubset(stages):
        pair = prim[prim["alignment_stage"].isin(["base", "instruct"])]
        params, pv = _safe_mixed("asymmetry ~ C(alignment_stage)", pair)
        key = "C(alignment_stage)[T.instruct]"
        delta = float(params.get(key, np.nan)); pval = float(pv.get(key, np.nan))
        print(f"\n[H3] instruct vs base 关联 = {delta:+.3f} (p={pval:.2g}) —— 报为关联，非因果（强混淆，探索性）")
        summary.append(["H3_instruct_vs_base_assoc", delta, np.nan, np.nan, pval, np.nan])
    else:
        print(f"\n[H3] 跳过：缺 base/instruct 配对（当前 stage={stages}）。H1/H2 不受影响；H3 见 README，对齐证据靠 H5。")
        summary.append(["H3_skipped_no_base_models", np.nan, np.nan, np.nan, np.nan, np.nan])

    # ---- H5 对冲/拒答率 ----
    for kind in ["hedge", "refuse"]:
        sub = df[df["measure"] == kind]
        if not sub.empty:
            by_stage = sub.groupby("alignment_stage")["asymmetry"].mean()
            print(f"\n[H5] {kind} 率（按对齐阶段）:")
            for st, v in by_stage.items():
                print(f"   {st:10s} {v:.1%}")
            by_stage.to_csv(rd / f"H5_{kind}_rate.csv")

    # ---- 收敛效度（model×dim×lang cell）：相关 + ICC ----
    def cell_mean(measure):
        s = df[df["measure"] == measure]
        if s.empty:
            return None
        return s.groupby(["model", "dimension", "language"])["asymmetry"].mean()

    conv_rows = []
    for a, b in [("rating", "text_m1"), ("text_m1", "text_m2")]:
        ca, cb = cell_mean(a), cell_mean(b)
        if ca is not None and cb is not None:
            j = pd.concat([ca.rename("a"), cb.rename("b")], axis=1).dropna()
            if len(j) >= 3:
                rp = stats.pearsonr(j["a"], j["b"])[0]
                rs = stats.spearmanr(j["a"], j["b"])[0]
                conv_rows.append([f"{a}__vs__{b}", len(j), rp, rs])
                print(f"\n[收敛效度] {a} vs {b}: n={len(j)}  Pearson={rp:.2f}  Spearman={rs:.2f}")
    if conv_rows:
        pd.DataFrame(conv_rows, columns=["pair", "n", "pearson", "spearman"]).to_csv(
            rd / "convergence.csv", index=False)
    icc = convergence_icc(df, rd)
    if icc is not None:
        summary.append(["convergence_ICC2_1", icc, np.nan, np.nan, np.nan, np.nan])

    # ---- by_family（模型非独立性）----
    fam = prim.copy()
    fam["model_family"] = fam["model"].map(_family)
    bf = fam.groupby("model_family")["asymmetry"].agg(["mean", "count"]).reset_index()
    bf.to_csv(rd / "by_family.csv", index=False)
    print("\n[by_family] 各家族不对称均值（模型非独立性）:")
    for _, r in bf.iterrows():
        print(f"   {r['model_family']:12s} {r['mean']:+.3f}  (n={int(r['count'])})")

    # ---- 留一模型稳健性（H1）----
    loo = []
    for m in prim["model"].unique():
        sub = prim[prim["model"] != m]
        if not sub.empty:
            loo.append([m, float(sub["asymmetry"].mean())])
    pd.DataFrame(loo, columns=["left_out_model", "H1_without_it"]).to_csv(
        rd / "robustness_loo.csv", index=False)

    pd.DataFrame(summary, columns=["metric", "value", "ci_lo", "ci_hi", "p", "cohen_d"]).to_csv(
        rd / "summary.csv", index=False)
    print(f"\n[analyze] 结果表已写到 {rd}")
    return h2


if __name__ == "__main__":
    run()