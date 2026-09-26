#!/usr/bin/env python3
"""Empty-response diagnosis by model x format (used during the June 2026 collection; unchanged).
Usage:  python -m tools.check_empty
"""
import os
import json
import glob
from collections import defaultdict
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src import config


def main():
    files = sorted(glob.glob(str(config.raw_dir() / "*.jsonl")))
    # model -> format -> {"total":n, "empty":n}
    stat = defaultdict(lambda: defaultdict(lambda: {"total": 0, "empty": 0}))
    skipped = []
    for path in files:
        base = os.path.basename(path)
        if base.startswith("mock") or base.startswith("_"):
            skipped.append(base)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                model = r.get("model", "?")
                fmt = r.get("format", "?")
                raw = r.get("raw_response", "")
                s = stat[model][fmt]
                s["total"] += 1
                if not raw or not raw.strip():
                    s["empty"] += 1

    if skipped:
        print(f"（已跳过 mock/辅助文件：{skipped}）\n")

    print("==== 空回复诊断（按模型 × 格式）====\n")
    for model in sorted(stat, key=lambda m: -sum(v["empty"] for v in stat[m].values())):
        fmts = stat[model]
        tot = sum(v["total"] for v in fmts.values())
        emp = sum(v["empty"] for v in fmts.values())
        if emp == 0:
            continue   # 这个模型完全健康，不占篇幅
        print(f"● {model}   总体空回复率 = {emp}/{tot} = {emp/max(tot,1):.1%}")
        for fmt, v in sorted(fmts.items(), key=lambda kv: -kv[1]["empty"]):
            if v["empty"] == 0:
                continue
            print(f"    {fmt:10s}  {v['empty']}/{v['total']}  ({v['empty']/max(v['total'],1):.1%})")
        # 判定
        only_rating = all(v["empty"] == 0 for fmt, v in fmts.items() if fmt != "rating")
        if only_rating and fmts.get("rating", {}).get("empty", 0) > 0:
            print("    👉 判定：空回复【只发生在 rating 格式】——很可能是 max_tokens 太小，"
                  "思考型模型把预算耗在隐藏推理上。建议：调大 generation.max_tokens.rating 后【只重跑这个模型】，零额外花费。")
        else:
            print("    👉 判定：空回复【跨多种格式】——这次调用本身就不稳定（限速/超时/该模型在 NIM 上欠佳）。"
                  "调 max_tokens 未必能救；建议先按【层1】方案小范围重试一次，仍大面积空 → 直接从主分析剔除该模型。")
        print()

    healthy = [m for m in stat if all(v["empty"] == 0 for v in stat[m].values())]
    if healthy:
        print(f"完全健康（零空回复）的模型：{healthy}")


if __name__ == "__main__":
    main()