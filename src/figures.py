"""出图：Fig2 分模型森林图、Fig3 维度梯度、Fig4 base vs instruct、Fig5 对冲率。
全部存到 data/results/*.png。"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import config

warnings.filterwarnings("ignore")
BLUE, RED, GREY = "#2b6cb0", "#c53030", "#718096"


def _primary(df):
    if (df["measure"] == "rating").any():
        return df[df["measure"] == "rating"]
    return df[df["measure"] == "text_m1"]


def run():
    rd = config.results_dir()
    df = pd.read_parquet(config.path("measured"))
    prim = _primary(df)

    # Fig2 分模型森林图
    g = prim.groupby("model")["asymmetry"].agg(["mean", "sem", "count"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(g))))
    y = np.arange(len(g))
    ax.errorbar(g["mean"], y, xerr=1.96 * g["sem"].fillna(0), fmt="o", color=BLUE, capsize=3)
    ax.axvline(0, color=GREY, ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(g.index, fontsize=8)
    ax.set_xlabel("Human–AI valence asymmetry  (>0 = machine-deference)")
    ax.set_title("Fig 2. Per-model asymmetry")
    plt.tight_layout(); plt.savefig(rd / "fig2_per_model.png", dpi=130); plt.close()

    # Fig3 维度梯度
    d = prim.groupby("dimension")["asymmetry"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [BLUE if v >= 0 else RED for v in d.values]
    ax.barh(d.index, d.values, color=colors)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_xlabel("Asymmetry  (blue >0 machine-favored, red <0 human-favored)")
    ax.set_title("Fig 3. Asymmetry by dimension (H2)")
    plt.tight_layout(); plt.savefig(rd / "fig3_by_dimension.png", dpi=130); plt.close()

    # Fig4 base vs instruct（若有）
    if {"base", "instruct"}.issubset(set(prim["alignment_stage"].unique())):
        s = prim[prim["alignment_stage"].isin(["base", "instruct"])]
        gg = s.groupby(["dimension", "alignment_stage"])["asymmetry"].mean().unstack()
        fig, ax = plt.subplots(figsize=(7, 4))
        gg.plot(kind="bar", ax=ax, color={"base": RED, "instruct": BLUE})
        ax.axhline(0, color=GREY, lw=1)
        ax.set_ylabel("Asymmetry"); ax.set_title("Fig 4. base vs instruct (H3, association)")
        ax.legend(title="")
        plt.tight_layout(); plt.savefig(rd / "fig4_base_vs_instruct.png", dpi=130); plt.close()

    # Fig5 对冲率
    h = df[df["measure"] == "hedge"]
    if not h.empty:
        hg = h.groupby("alignment_stage")["asymmetry"].mean()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(hg.index, hg.values, color=BLUE)
        ax.set_ylabel("Hedge rate"); ax.set_title("Fig 5. Hedging by alignment stage (H5)")
        plt.tight_layout(); plt.savefig(rd / "fig5_hedge_rate.png", dpi=130); plt.close()

    print(f"[figures] 图已写到 {rd}")


if __name__ == "__main__":
    run()
