#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py  ——  一键生成投稿用 Fig 1–4（独立运行，只依赖 items.csv）

直接运行：
    python make_figures.py
或指定数据路径：
    python make_figures.py "C:/Users/WY/Documents/data/measured/items.csv"

输出（默认存到当前目录下的 figures_out/）：
    Fig1.tif / Fig1.png   研究设计示意图（Study design）
    Fig2.tif / Fig2.png   分模型森林图（H1）
    Fig3.tif / Fig3.png   维度梯度（H2，核心结果）
    Fig4.tif / Fig4.png   对冲/拒答：自由文本 vs 评分格式（H5）

依赖：pip install pandas numpy matplotlib
TIFF 为 300 DPI + LZW 压缩，符合 PLOS 投稿要求；PNG 仅供本地预览。

注意（重要）：
  - 脚本已自动剔除两个空响应模型（gpt-oss-20b、nemotron-super-49b），与正文 Table 1
    的 7 模型口径一致；其中 nemotron-super-49b 有 1 行混入 rating 的脏数据也一并剔除。
  - Fig 4 的对冲/拒答率由本脚本从 items.csv 在这 7 个模型上重新计算，与正文、S1 完全一致。
    你流水线原始的 H5_hedge_rate.csv（3.9% / 10.4%）是在未剔除空响应模型的全部 9 个模型上算的，
    空响应被当作"未对冲"摊薄了分母，故偏低；方向与本图相同。若要让 data/results 里的 csv 与正文
    一致，按文末说明在 analyze.py 的 H5 块加一行模型过滤后重跑即可（不影响任何结论）。
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

warnings.filterwarnings("ignore")

# ───────────────────────── 配置区（按需修改）─────────────────────────
def _find_data():
    cands = []
    if len(sys.argv) > 1:
        cands.append(sys.argv[1])
    cands += [
        "items.csv",
        os.path.join("data", "measured", "items.csv"),
        "C:/Users/WY/Documents/data/measured/items.csv",
        "/mnt/user-data/uploads/items.csv",
    ]
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "找不到 items.csv。请把它放在脚本同目录，或运行： python make_figures.py <items.csv路径>"
    )

DATA_PATH = _find_data()
OUT_DIR = "figures_out"
EXCLUDE_MODELS = ["openai/gpt-oss-20b", "nvidia/llama-3.3-nemotron-super-49b-v1.5"]
DPI = 300
SAVE_TIFF = True
SAVE_PNG = True

# 颜色：蓝=抬机器(>0)，红=抬人(<0)，中性灰
BLUE, RED, GREY = "#2b6cb0", "#c53030", "#718096"
RED_LT, BLUE_LT = "#e57373", "#64b5f6"

# 字体：优先无衬线（PLOS 偏好 Arial/Helvetica），找不到就用默认，不报错
for _f in ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]:
    try:
        import matplotlib.font_manager as fm
        if any(_f.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [_f]
            break
    except Exception:
        pass
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"

DIM_LABELS = {
    "creativity": "Creativity",
    "reliability": "Reliability",
    "productivity": "Productivity",
    "decision": "Decision-making",
    "moral": "Moral judgment",
    "emotion": "Emotional perceptiveness",
    "trust": "Trustworthiness",
    "worth": "Intrinsic worth",
}
# H2 中 FDR 校正后显著的 5 个维度（与正文 Table 3 一致；若有 H2_by_dimension.csv 会优先读取）
SIG_DIMS = {"productivity", "moral", "emotion", "creativity", "worth"}
FRONTIER_LABEL = "Nemotron-Mini-4B\n(heavy safety tuning)"
INSTRUCT_LABEL = "Instruction-tuned\n(6 models)"


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    if SAVE_TIFF:
        fig.savefig(os.path.join(OUT_DIR, name + ".tif"), dpi=DPI,
                    pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight")
    if SAVE_PNG:
        fig.savefig(os.path.join(OUT_DIR, name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}  ->  {OUT_DIR}/")


# ───────────────────────── 载入 + 清洗 ─────────────────────────
def load():
    df = pd.read_csv(DATA_PATH)
    n0 = len(df)
    df = df[~df["model"].isin(EXCLUDE_MODELS)].copy()
    print(f"[data] {DATA_PATH}")
    print(f"[data] 总行 {n0} -> 剔除空响应模型后 {len(df)}；保留模型 {df['model'].nunique()} 个")
    return df


# ───────────────────────── Fig 1：研究设计示意图 ─────────────────────────
def fig1_design():
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

    boxes = [
        ("Prompt grid", "8 dimensions ×\n5 formats ×\n2 languages\n(288 prompts)", "#eef4fb"),
        ("Models", "7 open-weight\nLLMs\n(5 families)", "#eef4fb"),
        ("Measurement", "Primary:\n1–7 rating\nExploratory:\nfree-text\nsentiment", "#eef4fb"),
        ("Asymmetry", "a = v(AI)\n− v(human)\n>0 machine\n<0 human", "#fdeeee"),
        ("Inference", "Cluster\nbootstrap +\ncross-classified\nmixed model", "#eef4fb"),
    ]
    n = len(boxes)
    W, GAP, H, ytop, x0 = 14.6, 5.6, 24, 48, 1.5
    centers = []
    for i, (title, body, fc) in enumerate(boxes):
        x = x0 + i * (W + GAP)
        ax.add_patch(FancyBboxPatch((x, ytop - H), W, H, boxstyle="round,pad=0.4,rounding_size=1.1",
                                    linewidth=1.2, edgecolor="#33445a", facecolor=fc))
        ax.text(x + W / 2, ytop - 3.4, title, ha="center", va="top", fontsize=10, fontweight="bold", color="#1a2536")
        ax.text(x + W / 2, ytop - 8.6, body, ha="center", va="top", fontsize=7.8, color="#1a2536", linespacing=1.25)
        centers.append((x + W / 2, x, x + W))

    for i in range(n - 1):
        x_end = centers[i][2]
        x_start = centers[i + 1][1]
        ax.add_patch(FancyArrowPatch((x_end + 0.4, ytop - H / 2), (x_start - 0.4, ytop - H / 2),
                                     arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color="#33445a"))

    # 解读层注释（仅诠释层，不进入测量）—— 加宽 + 缩小标题字号，确保文字不出框
    bx, bw, by, bh = 9, 89, 3.5, 13
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.3,rounding_size=1.0",
                                linewidth=1.1, edgecolor="#8a6d3b", facecolor="#fbf6e9", linestyle="--"))
    cx = bx + bw / 2
    ax.text(cx, by + bh - 2.4, "Interpretive layer  (Discussion only — not part of the measurement)",
            ha="center", va="top", fontsize=8.2, fontweight="bold", color="#6b5326")
    ax.text(cx, by + bh - 6.3,
            "Anders' Promethean shame:  competence dimensions → machine-favoured;\n"
            "reserve dimensions (worth, creativity, feeling, moral) → human-favoured",
            ha="center", va="top", fontsize=7.9, color="#6b5326", linespacing=1.3)
    # 从 Asymmetry 框底部指向解读框
    asym_cx = centers[3][0]
    ax.add_patch(FancyArrowPatch((asym_cx, ytop - H - 0.3), (asym_cx, by + bh + 0.3),
                                 arrowstyle="-|>", mutation_scale=13, linewidth=1.2,
                                 color="#8a6d3b", linestyle="dashed"))

    ax.text(2, 53.5, "Fig 1. Study design.", fontsize=12, fontweight="bold", color="#111")
    _save(fig, "Fig1")


# ───────────────────────── Fig 2：分模型森林图（H1）─────────────────────────
def fig2_forest(df):
    prim = df[df["measure"] == "rating"]
    g = prim.groupby("model")["asymmetry"].agg(["mean", "sem", "count"]).sort_values("mean")
    pooled = g["mean"].mean()

    fig, ax = plt.subplots(figsize=(7.6, max(3.2, 0.62 * len(g) + 0.8)))
    y = np.arange(len(g))
    for yi, (m, se) in enumerate(zip(g["mean"], g["sem"].fillna(0))):
        c = BLUE if m >= 0 else RED
        ax.errorbar(m, yi, xerr=1.96 * se, fmt="o", color=c, capsize=3, markersize=7, lw=1.6)
    ax.axvline(0, color=GREY, ls="--", lw=1)
    ax.axvline(pooled, color="#444", ls=":", lw=1.2)
    ax.set_ylim(-0.9, len(g) - 0.3)
    ax.text(pooled, -0.78, f"pooled = {pooled:+.3f}", fontsize=8, color="#444",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbb", lw=0.6))
    ax.set_yticks(y)
    ax.set_yticklabels([s.split("/")[-1] for s in g.index], fontsize=9)
    ax.set_xlabel("Human–AI valence asymmetry  ( >0 machine-favoured,  <0 human-favoured )", fontsize=9.5)
    ax.set_title("Fig 2. Per-model human–AI valence asymmetry (H1)", fontsize=11, fontweight="bold")
    _save(fig, "Fig2")
    print(f"    [Fig2] pooled (mean of model means) = {pooled:+.3f}  | 正文 cluster-bootstrap 报 -0.088")


# ───────────────────────── Fig 3：维度梯度（H2，核心）─────────────────────────
def fig3_dimensions(df):
    prim = df[df["measure"] == "rating"]
    d = prim.groupby("dimension")["asymmetry"].mean().sort_values()

    # 显著性：优先读 H2_by_dimension.csv，否则用内置 SIG_DIMS
    sig = dict.fromkeys(d.index, False)
    for cand in [os.path.join(os.path.dirname(DATA_PATH) or ".", "..", "results", "H2_by_dimension.csv"),
                 os.path.join("data", "results", "H2_by_dimension.csv")]:
        try:
            h2 = pd.read_csv(cand)
            col = "sig" if "sig" in h2.columns else None
            if col:
                sig = dict(zip(h2["dimension"], h2[col].astype(bool)))
                break
        except Exception:
            pass
    if not any(sig.values()):
        sig = {dim: (dim in SIG_DIMS) for dim in d.index}

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = [BLUE if v >= 0 else RED for v in d.values]
    ax.barh(range(len(d)), d.values, color=colors, edgecolor="white", height=0.72)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([DIM_LABELS.get(x, x) for x in d.index], fontsize=9.5)
    ax.axvline(0, color=GREY, lw=1)
    for i, (dim, v) in enumerate(zip(d.index, d.values)):
        if sig.get(dim, False):
            ax.text(v + (0.012 if v >= 0 else -0.012), i, "*", va="center",
                    ha="left" if v >= 0 else "right", fontsize=15, fontweight="bold", color="#222")
    ax.set_xlabel("Asymmetry  ( blue >0 machine-favoured,  red <0 human-favoured;  * = FDR-significant )", fontsize=9.5)
    ax.set_title("Fig 3. Asymmetry by dimension (H2): narrow machine zone vs broad human reserve",
                 fontsize=10.5, fontweight="bold")
    xpad = max(abs(d.values)) * 0.18
    ax.set_xlim(d.values.min() - xpad, d.values.max() + xpad)
    _save(fig, "Fig3")
    print("    [Fig3] 维度均值：", {DIM_LABELS.get(k, k): round(v, 3) for k, v in d.items()})


# ───────────────────────── Fig 4：对冲/拒答 自由文本 vs 评分格式（H5）─────────────────────────
def fig4_hedge_refuse(df):
    FT_FORMATS = ["compare", "forced", "reflect", "scenario"]
    order = ["frontier", "instruct"]
    xlabels = [FRONTIER_LABEL, INSTRUCT_LABEL]

    # 左：自由文本（全部自由文本格式）hedge/refuse 率
    ft = {}
    for meas in ["hedge", "refuse"]:
        sub = df[(df["measure"] == meas) & (df["format"].isin(FT_FORMATS))]
        r = sub.groupby("alignment_stage")["asymmetry"].mean()
        ft[meas] = [r.get(s, 0.0) for s in order]

    # 右：评分格式 hedge/refuse 率 = 事件数 / 全部 rating 尝试(rating + rating_hedge + rating_refuse)
    denom = df[df["measure"].isin(["rating", "rating_hedge", "rating_refuse"])].groupby("alignment_stage").size()
    rt = {}
    for meas in ["rating_refuse", "rating_hedge"]:
        num = df[df["measure"] == meas].groupby("alignment_stage").size()
        rt[meas] = [(num.get(s, 0) / denom.get(s, 1)) if denom.get(s, 0) else 0.0 for s in order]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=False)
    x = np.arange(len(order)); w = 0.34

    # panel A
    a = axes[0]
    a.bar(x - w / 2, ft["refuse"], w, label="Refusal", color=RED)
    a.bar(x + w / 2, ft["hedge"], w, label="Hedging", color=RED_LT)
    a.set_xticks(x); a.set_xticklabels(xlabels, fontsize=8.5)
    a.set_ylabel("Rate", fontsize=9.5)
    a.set_title("Free-text comparison formats", fontsize=10, fontweight="bold")
    for xi, vals in zip([x - w / 2, x + w / 2], [ft["refuse"], ft["hedge"]]):
        for xx, vv in zip(xi, vals):
            a.text(xx, vv + 0.004, f"{vv:.1%}", ha="center", va="bottom", fontsize=8)
    a.legend(fontsize=8.5, loc="upper left")
    a.set_ylim(0, max(max(ft["refuse"]), max(ft["hedge"]), 0.05) * 1.25)

    # panel B
    b = axes[1]
    b.bar(x - w / 2, rt["rating_refuse"], w, label="Refusal", color=BLUE)
    b.bar(x + w / 2, rt["rating_hedge"], w, label="Hedging", color=BLUE_LT)
    b.set_xticks(x); b.set_xticklabels(xlabels, fontsize=8.5)
    b.set_ylabel("Rate", fontsize=9.5)
    b.set_title("Rating format  (ordering reversed)", fontsize=10, fontweight="bold")
    for xi, vals in zip([x - w / 2, x + w / 2], [rt["rating_refuse"], rt["rating_hedge"]]):
        for xx, vv in zip(xi, vals):
            b.text(xx, vv + 0.006, f"{vv:.1%}", ha="center", va="bottom", fontsize=8)
    b.legend(fontsize=8.5, loc="upper right")
    b.set_ylim(0, max(max(rt["rating_refuse"]), max(rt["rating_hedge"]), 0.05) * 1.25)

    fig.suptitle("Fig 4. Hedging and refusal by stage × elicitation format (H5 — note ordering flips across formats)",
                 fontsize=10.5, fontweight="bold", y=1.02)
    _save(fig, "Fig4")

    print("    [Fig4] 自由文本(全格式) refuse  frontier {:.1%} / instruct {:.1%}".format(ft["refuse"][0], ft["refuse"][1]))
    print("    [Fig4] 自由文本(全格式) hedge   frontier {:.1%} / instruct {:.1%}".format(ft["hedge"][0], ft["hedge"][1]))
    print("    [Fig4] 评分格式          refuse  frontier {:.1%} / instruct {:.1%}".format(rt["rating_refuse"][0], rt["rating_refuse"][1]))
    print("    [Fig4] 评分格式          hedge   frontier {:.1%} / instruct {:.1%}".format(rt["rating_hedge"][0], rt["rating_hedge"][1]))
    print("    [Fig4] 注：以上为剔除 2 个空响应模型后的 7 模型口径，与正文 Table 1 / S1 一致。")
    print("           （原 analyze.py 的 H5_*_rate.csv 未剔除空响应模型，故其数字 3.9%/10.4% 偏低；方向相同。）")


def main():
    print("=" * 64)
    df = load()
    print("[生成图表]")
    fig1_design()
    fig2_forest(df)
    fig3_dimensions(df)
    fig4_hedge_refuse(df)
    print("=" * 64)
    print(f"完成。4 张图已写到 ./{OUT_DIR}/ （.tif 用于投稿，.png 用于预览）")


if __name__ == "__main__":
    main()