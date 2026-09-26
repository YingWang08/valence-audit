"""Figures for the revised manuscript (PLOS ONE: TIFF, 300 dpi, LZW, width <= 7.5 in,
no titles or captions inside image files; captions live in the manuscript).

Fig1  study design
Fig2  per-model overall asymmetry (H1) with the equal-weight pooled estimate and its t(G-1) CI
Fig3  dimension-level asymmetry: (A) model-level means with t(G-1) 95% CIs and each model's value,
      (B) English vs Chinese
Fig4  rating-response outcomes by model and referent (missingness; Reviewer 1 #4, Reviewer 2 #2)
S1_Fig sensitivity analyses (dimension means under each specification)
(S2_Fig: revision-round referent profiles, written by src/r1.py)
Output: <data root>/results/figures/
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from src import config

BLUE, RED, GREY = "#2b6cb0", "#c53030", "#718096"
DIM_LABELS = {"creativity": "Creativity", "reliability": "Reliability", "productivity": "Productivity",
              "decision": "Decision-making", "moral": "Moral judgment", "emotion": "Emotional perceptiveness",
              "trust": "Trustworthiness", "worth": "Intrinsic worth"}
CAT_COLORS = {"valid": "#4a7c59", "valid_range": "#8fbc8f", "empty": "#bdbdbd", "refusal": "#c53030",
              "hedge": "#dd8452", "truncated": "#8172b3", "out_of_range": "#937860", "multiple": "#da8bc3",
              "malformed": "#555555"}

for _f in ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]:
    try:
        import matplotlib.font_manager as fm
        if any(_f.lower() == f.name.lower() for f in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [_f]
            break
    except Exception:
        pass
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False})


def _short(m):
    return str(m).split("/")[-1]


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + ".tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"},
                bbox_inches="tight")
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def fig1(outdir, n_models, n_families, n_prompts=288, n_rating=128):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    boxes = [("Prompt grid", f"8 dimensions x\n2 languages\n{n_rating} rating prompts\n(+{n_prompts - n_rating} free-text,\nnot analysed)"),
             ("Models", f"{n_models} instruction-\ntuned open-\nweight LLMs\n({n_families} families)"),
             ("Measurement", "1-7 rating of\n'a human' and\n'an AI system'\nin separate\nprompts"),
             ("Asymmetry", "a = [v(AI) -\nv(human)] / 6\nper cell\n>0 machine\n<0 human"),
             ("Inference", "Model-level\nmeans, t(G-1)\n+ sign-flip,\nwild bootstrap,\nfamily level")]
    W, GAP, H, ytop, x0 = 16.4, 3.9, 30, 46, 1.0
    xs = []
    for i, (title, body) in enumerate(boxes):
        x = x0 + i * (W + GAP)
        ax.add_patch(FancyBboxPatch((x, ytop - H), W, H, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    linewidth=1.0, edgecolor="#33445a", facecolor="#fdeeee" if i == 3 else "#eef4fb"))
        ax.text(x + W / 2, ytop - 2.6, title, ha="center", va="top", fontsize=8.5, fontweight="bold")
        ax.text(x + W / 2, ytop - 7.8, body, ha="center", va="top", fontsize=7.2, linespacing=1.25)
        xs.append((x, x + W))
    for i in range(len(boxes) - 1):
        ax.add_patch(FancyArrowPatch((xs[i][1] + 0.3, ytop - H / 2), (xs[i + 1][0] - 0.3, ytop - H / 2),
                                     arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#33445a"))
    ax.add_patch(FancyBboxPatch((8, 2), 84, 9, boxstyle="round,pad=0.3,rounding_size=1.0", linewidth=0.9,
                                edgecolor="#8a6d3b", facecolor="#fbf6e9", linestyle="--"))
    ax.text(50, 8.6, "Interpretive layer (Discussion only; not part of the measurement)", ha="center", va="top",
            fontsize=7.4, fontweight="bold", color="#6b5326")
    ax.text(50, 5.2, "Anders' Promethean shame offered as one reading of the observed dimension profile",
            ha="center", va="top", fontsize=7.2, color="#6b5326")
    _save(fig, outdir, "Fig1")


def fig2(outdir, rd):
    pm = pd.read_csv(rd / "H1_per_model.csv").sort_values("h1_model_mean")
    cells = pd.read_csv(rd / "cells_primary.csv")
    h1 = pd.read_csv(rd / "H1_overall.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(6.0, 0.42 * len(pm) + 1.4))
    for yi, (_, r) in enumerate(pm.iterrows()):
        v = cells[cells["model"] == r["model"]]["a"]
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
        c = BLUE if r["h1_model_mean"] >= 0 else RED
        ax.errorbar(r["h1_model_mean"], yi + 1, xerr=1.96 * se, fmt="o", color=c, capsize=2.5, ms=5, lw=1.2)
    m, lo, hi = h1["mean"], h1["ci_lo"], h1["ci_hi"]
    ax.fill([lo, m, hi, m], [0, 0.22, 0, -0.22], color="#333333")
    ax.axvline(0, color=GREY, ls="--", lw=0.9)
    ax.set_yticks(range(len(pm) + 1))
    ax.set_yticklabels([f"Pooled (equal weight), t({int(h1['G']) - 1})"] + [_short(x) for x in pm["model"]])
    ax.set_xlabel("Human-AI asymmetry a (>0 machine-favoured, <0 human-favoured)")
    _save(fig, outdir, "Fig2")


def fig3(outdir, rd):
    t3 = pd.read_csv(rd / "T3_H2_model_level.csv")
    md = pd.read_csv(rd / "S_model_by_dimension.csv", index_col=0)
    lang = pd.read_csv(rd / "S_language_by_dimension.csv")
    order = t3.sort_values("mean")["dimension"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    rng = np.random.default_rng(3)
    for i, d in enumerate(order):
        r = t3[t3["dimension"] == d].iloc[0]
        vals = md.loc[d].dropna().values
        ax.scatter(vals, i + rng.uniform(-0.18, 0.18, len(vals)), s=9, color="#999999", zorder=1)
        c = BLUE if r["mean"] >= 0 else RED
        ax.errorbar(r["mean"], i, xerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="D", color=c,
                    ms=5, capsize=2.5, lw=1.4, zorder=3)
        if r["p_t_BH"] < 0.05:
            ax.text(max(r["ci_hi"], vals.max()) + 0.03, i, "*", va="center", fontsize=11)
    ax.axvline(0, color=GREY, ls="--", lw=0.9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DIM_LABELS.get(d, d) for d in order])
    ax.set_xlabel("Asymmetry a (model-level mean, 95% CI)")
    ax.set_title("A", loc="left", fontweight="bold")
    ax = axes[1]
    for j, (lg, col, off) in enumerate((("language en", "#4c72b0", -0.14), ("language zh", "#dd8452", 0.14))):
        sub = lang[lang["spec"] == lg].set_index("dimension")
        for i, d in enumerate(order):
            if d not in sub.index:
                continue
            r = sub.loc[d]
            ax.errorbar(r["mean"], i + off, xerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="o",
                        color=col, ms=4, capsize=2, lw=1.1, label=("English" if lg.endswith("en") else "Chinese") if i == 0 else None)
    ax.axvline(0, color=GREY, ls="--", lw=0.9)
    ax.set_xlabel("Asymmetry a by prompt language")
    ax.set_title("B", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, outdir, "Fig3")


def fig4(outdir, rd):
    """Outcome composition by model x referent, separately for English and Chinese prompts."""
    rr = pd.read_csv(config.path("rating_responses"), usecols=["model", "agent", "language", "category"])
    cats = [c for c in CAT_COLORS if c in set(rr["category"])]
    valid_share = rr.assign(v=rr["category"].isin(["valid", "valid_range"])).groupby("model")["v"].mean()
    models = valid_share.sort_values().index.tolist()
    rows = [(m, ag) for m in models for ag in ("human", "ai")]
    ypos, y = [], 0.0
    for i, (m, ag) in enumerate(rows):
        ypos.append(y)
        y += 1.0 if ag == "human" else 1.45
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 0.34 * len(rows) + 1.3), sharey=True)
    for ax, lang, title in zip(axes, ("en", "zh"), ("A  English prompts", "B  Chinese prompts")):
        sub = rr[rr["language"] == lang]
        tab = pd.crosstab([sub["model"], sub["agent"]], sub["category"], normalize="index").reindex(columns=cats, fill_value=0) * 100
        for (m, ag), yy in zip(rows, ypos):
            if (m, ag) not in tab.index:
                continue
            left = 0.0
            for c in cats:
                w = tab.loc[(m, ag), c]
                ax.barh(yy, w, left=left, color=CAT_COLORS[c], edgecolor="white", linewidth=0.4, height=0.85,
                        label=c.replace("_", " ") if (m, ag) == rows[0] else None)
                left += w
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of rating responses (%)")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8.5)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([f"{_short(m)} | {'human' if ag == 'human' else 'AI system'}" for m, ag in rows], fontsize=6.8)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(l), frameon=False, fontsize=6.8, loc="lower center", bbox_to_anchor=(0.55, -0.04))
    fig.subplots_adjust(bottom=0.14)
    _save(fig, outdir, "Fig4")


def s_figs(outdir, rd):
    if config.EXP.get("analysis", {}).get("report_freetext", False) and (rd / "H5_by_model.csv").exists():
        _h5_fig(outdir, rd)
    _sensitivity_fig(outdir, rd)


def _h5_fig(outdir, rd):
    h5 = pd.read_csv(rd / "H5_by_model.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for ax, fmt in zip(axes, ["free-text (all)", "rating"]):
        sub = h5[h5["format"] == fmt].sort_values("model")
        y = np.arange(len(sub))
        ax.barh(y - 0.2, sub["hedge_pct"], height=0.38, color="#dd8452", label="hedge")
        ax.barh(y + 0.2, sub["refuse_pct"], height=0.38, color="#c53030", label="refusal")
        ax.set_yticks(y)
        ax.set_yticklabels([_short(m) + (" †" if s == "frontier" else "") for m, s in zip(sub["model"], sub["stage"])])
        ax.set_xlabel(f"% of {fmt} responses")
        ax.set_title("A" if fmt.startswith("free") else "B", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)
    _save(fig, outdir, "S_H5_Fig_exploratory")


def _sensitivity_fig(outdir, rd):
    sl = pd.read_csv(rd / "S_sensitivity_long.csv")
    sl = sl[sl["dimension"] != "H1_overall"]
    specs = list(dict.fromkeys(sl["spec"]))
    dims = pd.read_csv(rd / "T3_H2_model_level.csv").sort_values("mean")["dimension"].tolist()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    cmap = plt.get_cmap("tab20")
    for j, sp in enumerate(specs):
        sub = sl[sl["spec"] == sp].set_index("dimension")
        off = (j - len(specs) / 2) * 0.05
        ys = [i + off for i, d in enumerate(dims) if d in sub.index]
        xs = [sub.loc[d, "mean"] for d in dims if d in sub.index]
        ax.scatter(xs, ys, s=12 if sp != "primary" else 30, color="black" if sp == "primary" else cmap(j % 20),
                   marker="D" if sp == "primary" else "o", label=sp, zorder=3 if sp == "primary" else 2)
    ax.axvline(0, color=GREY, ls="--", lw=0.9)
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([DIM_LABELS.get(d, d) for d in dims])
    ax.set_xlabel("Model-level mean asymmetry a")
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.4, -0.12))
    _save(fig, outdir, "S1_Fig")


def run():
    rd = config.results_dir()
    outdir = rd / "figures"
    pm = pd.read_csv(rd / "H1_per_model.csv")
    n_rating = 128
    gp = config.path("prompts")
    n_prompts = 288
    if os.path.exists(gp):
        g = pd.read_json(gp, lines=True)
        n_prompts, n_rating = len(g), int((g["format"] == "rating").sum())
    for stale in ("S2_Fig.tif", "S2_Fig.png", "S_H5_Fig_exploratory.tif", "S_H5_Fig_exploratory.png"):
        (outdir / stale).unlink(missing_ok=True)
    fig1(outdir, pm["model"].nunique(), pm["family"].nunique(), n_prompts, n_rating)
    fig2(outdir, rd)
    fig3(outdir, rd)
    fig4(outdir, rd)
    s_figs(outdir, rd)
    print(f"[figures] written to {outdir}")


if __name__ == "__main__":
    run()
