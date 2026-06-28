#!/usr/bin/env python3
"""诊断 rating 解析失败。

用法： python check_parse.py            # 用 src/measure.py 里【当前的】_parse_rating 扫描 data/raw/
       python check_parse.py 10        # 每个模型最多打印 10 条失败样本（默认 6）

它回答三个问题：
  1) 2578 条解析失败集中在哪些模型？（思考型？文字数字？被 max_tokens 截断？）
  2) 这些模型失败时到底回了什么？（看失败样本原文）
  3) 解析出来的值是否合理？（看每个模型的 1-7 值分布，识别"恒为 1"=量纲回显没去干净 之类）

请先把新版 measure.py 放进 src/ 再运行本脚本（它直接 import src.measure._parse_rating，
这样诊断反映的是修复后的解析器，看残余失败是否还值得进一步处理）。
"""
import os
import sys
import json
import glob
from collections import defaultdict, Counter

from src import config
from src.measure import _parse_rating       # 用当前安装的解析器

N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def main():
    files = sorted(glob.glob(str(config.raw_dir() / "*.jsonl")))
    stat = defaultdict(lambda: {"total": 0, "ok": 0, "fail": 0,
                                "vals": Counter(), "samples": []})
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
                if r.get("format") != "rating":
                    continue
                model = r["model"]
                s = stat[model]
                s["total"] += 1
                raw = r.get("raw_response", "")
                v = _parse_rating(raw)
                if v is None:
                    s["fail"] += 1
                    if len(s["samples"]) < N_SAMPLE:
                        snippet = (raw or "").replace("\n", " ⏎ ").strip()
                        s["samples"].append(snippet[:160])
                else:
                    s["ok"] += 1
                    s["vals"][v] += 1

    if skipped:
        print(f"（已跳过 mock/辅助文件：{skipped}）\n")

    order = sorted(stat.items(), key=lambda kv: kv[1]["fail"], reverse=True)
    tot_total = sum(s["total"] for _, s in order)
    tot_fail = sum(s["fail"] for _, s in order)
    print(f"==== rating 解析诊断（共 {tot_total} 条，失败 {tot_fail}，失败率 {tot_fail / max(tot_total,1):.1%}）====\n")

    for model, s in order:
        rate = s["fail"] / max(s["total"], 1)
        dist = " ".join(f"{k}:{s['vals'][k]}" for k in sorted(s["vals"]))
        print(f"● {model}")
        print(f"    total={s['total']}  ok={s['ok']}  fail={s['fail']}  失败率={rate:.1%}")
        print(f"    解析值分布 [1-7]:  {dist or '(无)'}")
        if s["samples"]:
            print(f"    失败样本（最多 {N_SAMPLE} 条原文）:")
            for i, smp in enumerate(s["samples"], 1):
                print(f"      {i}. {smp!r}")
        print()

    print("解读提示：")
    print("  · 某模型失败率高 + 样本里有大段推理/<think> -> 思考型，建议 max_tokens 调大或从 rating 主分析剔除（保留它在 H5）。")
    print("  · 样本里是 'I would rate it as...'(被截断没数字) -> max_tokens=12 太小，rating 维度可单独把 max_tokens 调到 ~24 重跑。")
    print("  · 样本里是文字数字/奇怪格式而新解析器仍漏 -> 把样本发我，我再加规则。")
    print("  · 解析值分布 '恒为某数' 或全在 1 -> 可能量纲/锚点回显没去干净，需针对性修。")


if __name__ == "__main__":
    main()