"""生成阶段：对每个模型、每条 prompt、每次重复，调用模型并落盘。
特性：断点续跑（跳过已完成）、并发、限速、模型不可用自动跳过、用量日志。
输出：data/raw/<model>.jsonl
"""
import json
import csv
import time
import asyncio
import pathlib
from src import config
from src.providers import get_provider, ModelUnavailable, CallFailed


def _load_prompts(filter_dims=None):
    prompts = []
    with open(config.path("prompts"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if filter_dims and r["dimension"] not in filter_dims:
                continue
            prompts.append(r)
    return prompts


def _done_keys(out_path):
    done = set()
    if pathlib.Path(out_path).exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["prompt_id"], r["repeat"]))
                except Exception:
                    pass
    return done


def _guard_against_mock_contamination():
    """防止 data/raw/ 里残留的 --mock 假数据被悄悄混进真实分析。
    mock 模式硬编码模型名为 mock/llama-base、mock/llama-instruct（见上方 if mock 分支），
    文件名落地后是 mock_llama-base.jsonl / mock_llama-instruct.jsonl。
    真实（非 mock）运行前若发现这些文件残留，直接中止并提示清理——
    宁可多一步手动确认，也不要让分析结果掺进假数据却让人毫无察觉。"""
    leftovers = sorted(p.name for p in config.raw_dir().glob("mock_*.jsonl"))
    if leftovers:
        names = "、".join(leftovers)
        raise SystemExit(
            f"\n⚠️  检测到 data/raw/ 下有 --mock 残留的假数据文件：{names}\n"
            f"   继续跑真实数据会让 measure.py 把它们也读进去，悄悄污染分析结果（H1/H2/H3/H5 全部不可信）。\n"
            f"   请先清理后重跑，例如：\n"
            f"     rm -rf data        # 最彻底，连同 prompts/measured/results 一起重建（推荐）\n"
            f"   或只删 mock 残留：\n"
            f"     rm data/raw/mock_*.jsonl\n"
        )


async def _run_one_model(provider, model, stage, prompts, repeats, out_path, cost_writer):
    done = _done_keys(out_path)
    gen = config.EXP["generation"]
    sem = asyncio.Semaphore(config.EXP["api"]["concurrency"])
    fout = open(out_path, "a", encoding="utf-8")
    skip_model = {"flag": False}

    total = len(prompts) * repeats
    already_done = sum(1 for p in prompts for rep in range(repeats) if (p["prompt_id"], rep) in done)
    # 进度心跳：每完成约 1/10 总量打印一次，且不少于 50 次一报，避免刷屏也避免长时间静默
    heartbeat_every = max(50, total // 10)
    progress = {"n": 0, "ok": 0, "fail": 0, "skip_done": already_done, "t0": time.monotonic()}

    def _tick(kind):
        progress["n"] += 1
        progress[kind] = progress.get(kind, 0) + 1
        if progress["n"] % heartbeat_every == 0 or progress["n"] == total:
            elapsed = time.monotonic() - progress["t0"]
            done_total = progress["skip_done"] + progress["n"]
            rate = progress["n"] / elapsed if elapsed > 0 else 0
            remain = (total - done_total) / rate if rate > 0 else float("inf")
            print(f"     [进度] {model}: {done_total}/{total}"
                  f"（本次新完成 {progress['n']}，成功 {progress['ok']}，失败 {progress['fail']}）"
                  f"  已用时 {elapsed/60:.1f} 分钟  预计剩余 ~{remain/60:.1f} 分钟")

    async def one(p, rep):
        if skip_model["flag"]:
            return
        if (p["prompt_id"], rep) in done:
            return
        mt = gen["max_tokens"]["rating"] if p["format"] == "rating" else gen["max_tokens"]["other"]
        async with sem:
            try:
                text, usage = await provider.call(model, p["text"], gen["temperature"], mt, seed_hint=rep)
            except ModelUnavailable:
                skip_model["flag"] = True
                print(f"  [跳过] 模型不可用：{model}")
                return
            except CallFailed as e:
                print(f"  [失败] {model} 一条调用最终失败：{str(e)[:80]}")
                _tick("fail")
                return
        rec = {**p, "model": model, "alignment_stage": stage,
               "temperature": gen["temperature"], "repeat": rep, "raw_response": text}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        if cost_writer:
            cost_writer.writerow([model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)])
        _tick("ok")

    if already_done:
        print(f"     [续跑] {model}: 已有 {already_done}/{total} 条历史记录，将跳过这些、只补剩余的。")
    tasks = [one(p, rep) for p in prompts for rep in range(repeats)]
    # 先探测一条，模型不可用则整模型跳过，避免无谓并发
    if tasks:
        await tasks[0]
        if not skip_model["flag"]:
            await asyncio.gather(*tasks[1:])
    fout.close()
    return skip_model["flag"]


async def run(mock=False, smoke=False):
    if not mock:
        _guard_against_mock_contamination()          # 先拦截污染，再连接 API
    provider = get_provider(mock=mock)
    models = config.MODELS
    gen = config.EXP["generation"]
    repeats_map = gen["repeats"]

    if mock:
        models = [{"name": "mock/llama-base", "stage": "base"},
                  {"name": "mock/llama-instruct", "stage": "instruct"}]
    if smoke:
        sk = config.EXP["smoke"]
        models = models[:sk["models"]]
        prompts = _load_prompts(filter_dims=set(sk["dimensions"]))
        repeats_override = sk["repeats"]
    else:
        prompts = _load_prompts()
        repeats_override = None

    print(f"[generate] 模型数={len(models)}  prompt 数={len(prompts)}  mock={mock} smoke={smoke}")
    cost_path = config.path("cost_log")
    cost_f = open(cost_path, "a", newline="")
    cost_writer = csv.writer(cost_f)

    used = 0
    for m in models:
        stage = m["stage"]
        repeats = repeats_override if repeats_override is not None else repeats_map.get(stage, 5)
        out_path = config.raw_dir() / (m["name"].replace("/", "_") + ".jsonl")
        print(f"  -> {m['name']} (stage={stage}, repeats={repeats})")
        skipped = await _run_one_model(provider, m["name"], stage, prompts, repeats, out_path, cost_writer)
        if not skipped:
            used += 1
    cost_f.close()
    print(f"[generate] 完成。成功使用 {used}/{len(models)} 个模型。原始数据在 {config.raw_dir()}")
    if used == 0 and not mock:
        print("  ⚠️ 没有任何模型成功。先运行 `python run_all.py --list-models` 看可用模型，再改 config/models.yaml。")


if __name__ == "__main__":
    asyncio.run(run())