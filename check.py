import glob, os
for p in sorted(glob.glob("data/raw/*.jsonl")):
    n = sum(1 for _ in open(p, encoding="utf-8"))
    print(f"{os.path.basename(p):55s} {n:6d} 行")