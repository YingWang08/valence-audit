import os, glob, shutil
RAW, EXCLUDED = "data/raw", "data/raw_excluded"
os.makedirs(EXCLUDED, exist_ok=True)
KEEP_MIN = 1440   # frontier满额1440 / instruct满额2304；低于此=半成品
for p in glob.glob(os.path.join(RAW, "*.jsonl")):
    name = os.path.basename(p)
    n = sum(1 for _ in open(p, encoding="utf-8"))
    reason = "mock假数据" if name.startswith("mock_") else (f"半成品({n}行)" if n < KEEP_MIN else None)
    if reason:
        shutil.move(p, os.path.join(EXCLUDED, name)); print(f"移出 {name:50s} [{reason}]")
print("\n保留(将纳入分析):")
for p in sorted(glob.glob(os.path.join(RAW, "*.jsonl"))):
    print("  ", os.path.basename(p), sum(1 for _ in open(p, encoding="utf-8")), "行")