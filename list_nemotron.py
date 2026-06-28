import asyncio, re
from src.providers import NimProvider

ids = asyncio.run(NimProvider().list_models())

# 只看对齐强化(Nemotron)系，以及任何带 nano/mini/small 的小模型
cand = [m for m in ids if "nemotron" in m.lower()
        or any(k in m.lower() for k in ["nano", "mini", "small"])]

def size_key(m):                       # 粗略按参数量排序，小的在前
    g = re.search(r"(\d+)\s*b", m.lower())
    return int(g.group(1)) if g else 999

print(f"对齐强化 / 小尺寸候选（共 {len(cand)} 个，按尺寸粗排）:")
for m in sorted(cand, key=size_key):
    g = re.search(r"(\d+)\s*b", m.lower())
    print(f"  {m:55s} ~{g.group(1)+'B' if g else '?'}")