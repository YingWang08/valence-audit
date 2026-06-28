#!/usr/bin/env python3
"""一键运行入口。

用法：
  python run_all.py --mock         # 离线干跑（不需密钥/网络），验证整条流水线能跑通
  python run_all.py --list-models  # 列出你的 NVIDIA 密钥实际能调的模型（关键！先看有没有 base）
  python run_all.py --smoke        # 真实 API，1 模型 1 维度，最省额度，验证 key+连通
  python run_all.py --full         # 跑完整实验
  python run_all.py                # 无参数 = 打印本帮助
"""
import sys
import asyncio
from src import build_prompts, generate, measure, analyze, figures


def pipeline(mock=False, smoke=False):
    print("\n[1/5] 构建 prompt 网格 ...")
    build_prompts.build()
    print("\n[2/5] 生成 ...")
    asyncio.run(generate.run(mock=mock, smoke=smoke))
    print("\n[3/5] 测量 ...")
    measure.measure()
    print("\n[4/5] 分析 ...")
    analyze.run()
    print("\n[5/5] 出图 ...")
    figures.run()
    print("\n✅ 全部完成。结果在 data/results/（CSV 表 + PNG 图）。")


def list_models():
    from src.providers import NimProvider
    prov = NimProvider()
    ids = asyncio.run(prov.list_models())
    print(f"\n你的密钥可调用 {len(ids)} 个模型：")
    for i in sorted(ids):
        print("  -", i)
    print("\n👉 看里面有没有 base 模型（如 ...-base、不带 -instruct 的）。")
    print("   有 -> 填进 config/models.yaml 的 base 段，H3 完整可做。")
    print("   没有 -> H3 退化为探索性分析（见 README），H1/H2 不受影响。")


def main():
    args = set(sys.argv[1:])
    if "--mock" in args:
        pipeline(mock=True)
    elif "--list-models" in args:
        list_models()
    elif "--smoke" in args:
        pipeline(smoke=True)
    elif "--full" in args:
        pipeline()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
