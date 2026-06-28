#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regenerate_H5_csv.py  ——  把 H5_hedge_rate.csv / H5_refuse_rate.csv 重算成与正文一致的 7 模型口径。

背景：原 analyze.py 的 H5 块用全部 9 个模型算对冲/拒答率（没有剔除两个空响应模型
gpt-oss-20b、nemotron-super-49b）。这两个模型大量返回空响应，被当作"未对冲/未拒答"摊薄了
分母，使 frontier 的率偏低（hedge 3.9%、refuse 1.8%）。正文 Table 1/S1/图都用 7 个保留模型，
为内部一致，这里在同样的 7 个模型上重算。结论方向不变（frontier 在自由文本里仍更少对冲/拒答）。

运行：
    python regenerate_H5_csv.py                      # 自动找 items.csv，写到 data/results/
    python regenerate_H5_csv.py <items.csv> <输出目录>

—— 等价的"治本"改法：在 analyze.py 的 H5 循环里，把
        sub = df[df["measure"] == kind]
   改成
        sub = df[(df["measure"] == kind) & (~df["model"].isin(EXCLUDE_MODELS))]
   （EXCLUDE_MODELS 见下），重跑 analyze.run() 即可。H1/H2 用的 rating 指标几乎不受影响
   （那两个模型几乎没有有效 rating 行），但保持一致写法更稳妥。
"""
import os
import sys
import pandas as pd

EXCLUDE_MODELS = ["openai/gpt-oss-20b", "nvidia/llama-3.3-nemotron-super-49b-v1.5"]


def find_items():
    if len(sys.argv) > 1:
        return sys.argv[1]
    for c in ["items.csv", "data/measured/items.csv",
              "C:/Users/WY/Documents/data/measured/items.csv"]:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("找不到 items.csv，请作为第一个参数传入路径。")


def main():
    items = find_items()
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/results"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(items)
    d = df[~df["model"].isin(EXCLUDE_MODELS)].copy()
    print(f"[H5] 读 {items}；剔除空响应模型后保留 {d['model'].nunique()} 个模型")

    for kind in ["hedge", "refuse"]:
        sub = d[d["measure"] == kind]
        by_stage = sub.groupby("alignment_stage")["asymmetry"].mean()
        path = os.path.join(out_dir, f"H5_{kind}_rate.csv")
        by_stage.to_csv(path)
        rates = ", ".join(f"{k} {v:.1%}" for k, v in by_stage.items())
        print(f"[H5] {kind:7s} -> {path}   ({rates})")

    print("\n完成。这两个 csv 现在与正文 Fig 4 / Results 的 7 模型数字一致。")


if __name__ == "__main__":
    main()
