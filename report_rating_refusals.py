#!/usr/bin/env python3
"""独立报告：新增的 rating_hedge / rating_refuse 信号（零成本，纯本地读取 items.parquet/csv）。

不依赖、不修改你现有的 src/analyze.py —— 这是一份单独的查看脚本，先让你看清楚
"层2"救回来的信号长什么样，再决定要不要把它并入正式的 H5 主统计。

用法： python report_rating_refusals.py
"""
import pandas as pd
from src import config


def main():
    df = pd.read_parquet(config.path("measured"))

    new_measures = ["rating_hedge", "rating_refuse"]
    sub = df[df["measure"].isin(new_measures)]
    if sub.empty:
        print("没有发现 rating_hedge / rating_refuse 行。"
              "确认你用的是升级版 measure.py 重新跑过 measure 阶段（python -m src.measure）。")
        return

    print("==== 新增信号：rating 格式下的真实拒答/对冲（这次从'解析失败'里救回来的）====\n")

    print("【按模型】出现次数（也能看出哪些模型在 rating 格式下特别爱拒答）：")
    by_model = sub.groupby(["model", "measure"]).size().unstack(fill_value=0)
    print(by_model.to_string())
    print()

    print("【按对齐阶段】rating_hedge / rating_refuse 占该阶段全部 rating 尝试的比例：")
    rating_attempts = df[df["measure"].isin(["rating", "rating_hedge", "rating_refuse"])]
    for kind in new_measures:
        denom = rating_attempts.groupby("alignment_stage").size()
        numer = rating_attempts[rating_attempts["measure"] == kind].groupby("alignment_stage").size()
        rate = (numer / denom).fillna(0)
        print(f"\n  {kind}:")
        for stage, v in rate.items():
            print(f"    {stage:10s}  {v:.1%}   (n={int(numer.get(stage, 0))}/{int(denom.get(stage, 0))})")

    print("\n【对比】原有自由文本 hedge/refuse（forced/compare/reflect/scenario 格式）按对齐阶段：")
    old = df[df["measure"].isin(["hedge", "refuse"])]
    for kind in ["hedge", "refuse"]:
        s = old[old["measure"] == kind]
        if s.empty:
            continue
        rate = s.groupby("alignment_stage")["asymmetry"].mean()
        print(f"\n  {kind}（自由文本）:")
        for stage, v in rate.items():
            print(f"    {stage:10s}  {v:.1%}")

    print("\n──────────────────────────────────────────────")
    print("怎么用这份报告做决定：")
    print("  · 如果 rating_hedge/refuse 按对齐阶段的模式【和】自由文本 hedge/refuse 的模式【一致】")
    print("    （比如都是 instruct > frontier，或都是 frontier > instruct）")
    print("    -> 两个独立测量互相印证，H5 故事更稳，可以考虑在 analyze.py 里把它们【合并】统计，")
    print("       在 Methods 写明'对冲/拒答在评分式与开放式提示下一致定义并合并统计'。")
    print("  · 如果两者模式【不一致】（比如自由文本说 instruct 更爱对冲，rating 却说 frontier 更爱对冲）")
    print("    -> 不要硬合并去'凑'一个故事。分开报告，如实讨论'对冲行为可能依赖提示格式'，")
    print("       这本身也是一个值得写进 Discussion 的发现，比强行统一更诚实、更经得起审稿。")
    print("  · 这份新信号不需要再调用任何 API —— 它是从你已经收集到的 data/raw 里挖出来的，零成本。")


if __name__ == "__main__":
    main()