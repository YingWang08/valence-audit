#!/usr/bin/env python3
"""把空/空白 raw_response 从 data/raw/*.jsonl 隔离出去（零成本，不调用任何 API）。

原理：generate.py 的断点续跑靠 `_done_keys()` 读取 jsonl 里已有的 (prompt_id, repeat)；
凡是已经写进文件的行，无论内容是不是空字符串，都会被当成"已完成"而跳过，不会重试。
本脚本把空回复的行【移出】jsonl（搬到 data/raw_quarantine/ 备查），这样下次跑
`python run_all.py --full` 时，generate.py 会发现这些 (prompt_id, repeat) 还没做，
自动【只】重新调用这些坑——不影响任何已经成功拿到内容的样本，不浪费额度。

用法：
  python quarantine_empty.py                # 只看会动多少行（dry-run，不改任何文件）
  python quarantine_empty.py --apply         # 真正执行隔离
  python quarantine_empty.py --apply --model meta/llama-3.1-8b-instruct   # 只处理指定模型

建议流程：
  1) 先 dry-run 看一眼数字是否符合预期（应该约等于 check_empty.py 报告的空回复数）。
  2) 改 config/experiment.yaml：generation.max_tokens.rating 从 12 调到 40（仍是免费额度，零花费）。
  3) 再 --apply 真正隔离。
  4) python run_all.py --full   —— 断点续跑会自动只补这些坑。
"""
import os
import sys
import json
import glob
import shutil
from src import config


def main():
    apply = "--apply" in sys.argv
    only_model = None
    if "--model" in sys.argv:
        only_model = sys.argv[sys.argv.index("--model") + 1]

    raw = config.raw_dir()
    qdir = raw.parent / "raw_quarantine"
    qdir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(raw / "*.jsonl")))
    total_kept, total_removed = 0, 0

    for path in files:
        base = os.path.basename(path)
        if base.startswith("mock") or base.startswith("_"):
            continue

        kept_lines, removed_lines = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    kept_lines.append(line)   # 解析不了的行原样保留，不动它
                    continue
                model = r.get("model", "")
                if only_model and model != only_model:
                    kept_lines.append(line)
                    continue
                raw_resp = r.get("raw_response", "")
                if raw_resp is None or not str(raw_resp).strip():
                    removed_lines.append(line)
                else:
                    kept_lines.append(line)

        if not removed_lines:
            continue

        model_name = base.replace(".jsonl", "")
        print(f"● {model_name}: 总行数={len(kept_lines)+len(removed_lines)}  "
              f"将隔离空回复={len(removed_lines)}  保留={len(kept_lines)}")
        total_kept += len(kept_lines)
        total_removed += len(removed_lines)

        if apply:
            qpath = qdir / (model_name + ".empty.jsonl")
            with open(qpath, "a", encoding="utf-8") as qf:
                qf.writelines(removed_lines)
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(kept_lines)
            print(f"    已隔离 -> {qpath}；原文件已重写（少了这 {len(removed_lines)} 行）")

    print(f"\n{'[已执行]' if apply else '[DRY-RUN，未改任何文件，加 --apply 才真正执行]'} "
          f"共 {total_removed} 行将被隔离，{total_kept} 行保留。")
    if not apply and total_removed > 0:
        print("确认数字符合预期后，重新加 --apply 执行。")
    if apply and total_removed > 0:
        print("下一步：确认 experiment.yaml 的 generation.max_tokens.rating 已调大，"
              "然后跑 `python run_all.py --full`（断点续跑只会补这些坑，零浪费额度）。")


if __name__ == "__main__":
    main()