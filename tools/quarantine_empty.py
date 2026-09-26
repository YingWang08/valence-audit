#!/usr/bin/env python3
"""Move empty responses out of data/raw/*.jsonl into data/raw_quarantine/ so that a resumed
run re-queries only those (prompt, repeat) positions. This is the procedure used in June 2026
for the two excluded reasoning models. The token log shows that the re-query used the same
budgets as the first pass (rating 12, free text 256), so it could not rescue those models.
Usage:  python -m tools.quarantine_empty [--apply] [--model MODEL_ID]
"""
import os
import sys
import json
import glob
import shutil
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
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