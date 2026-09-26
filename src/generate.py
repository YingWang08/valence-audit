"""Generation: query every model with every prompt and repetition, append to JSONL.
Resumable (skips (prompt_id, repeat) already on disk), rate-limited, concurrent.

Each record now stores the generation parameters actually used (max_tokens,
temperature, system prompt flag), the API's finish_reason, token usage and the
length of any separate reasoning channel, plus a UTC timestamp and a collection
id. v1.0.0 stored none of these per record, which is why the June data can only
be screened for truncation heuristically (see tools/diagnose_usage.py).

Output: <data root>/raw/<model>.jsonl
"""
import json
import csv
import time
import asyncio
import datetime as dt
import pathlib
from src import config
from src.providers import get_provider, ModelUnavailable, CallFailed


def load_prompts(grid_path, filter_dims=None, formats=None):
    prompts = []
    with open(grid_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if filter_dims and r["dimension"] not in filter_dims:
                continue
            if formats and r["format"] not in formats:
                continue
            prompts.append(r)
    return prompts


def _done_keys(out_path):
    done = set()
    p = pathlib.Path(out_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["prompt_id"], r["repeat"]))
                except Exception:
                    pass
    return done


def _max_tokens_for(fmt, gen, override):
    table = dict(gen["max_tokens"])
    if override and "max_tokens" in override:
        mt = override["max_tokens"]
        if isinstance(mt, dict):
            table.update(mt)
        else:
            return int(mt)
    if fmt in table:
        return int(table[fmt])
    return int(table["rating"] if fmt == "rating" else table.get("other", 256))


async def _run_one_model(provider, model, stage, prompts, repeats, out_path, cost_writer,
                         gen, override, collection):
    done = _done_keys(out_path)
    sem = asyncio.Semaphore(config.EXP["api"]["concurrency"])
    fout = open(out_path, "a", encoding="utf-8")
    state = {"skip": False, "n": 0, "ok": 0, "fail": 0, "t0": time.monotonic()}
    total = len(prompts) * repeats
    already = sum(1 for p in prompts for rep in range(repeats) if (p["prompt_id"], rep) in done)
    beat = max(50, total // 10)
    system_prompt = (override or {}).get("system_prompt")
    extra_body = (override or {}).get("extra_body")

    def tick(kind):
        state["n"] += 1
        state[kind] += 1
        if state["n"] % beat == 0:
            el = time.monotonic() - state["t0"]
            print(f"     [progress] {model}: {already + state['n']}/{total} "
                  f"(ok {state['ok']}, failed {state['fail']}), {el / 60:.1f} min")

    async def one(p, rep):
        if state["skip"] or (p["prompt_id"], rep) in done:
            return
        mt = _max_tokens_for(p["format"], gen, override)
        async with sem:
            try:
                text, meta = await provider.call(model, p["text"], gen["temperature"], mt, seed_hint=rep,
                                                 system_prompt=system_prompt, extra_body=extra_body)
            except ModelUnavailable:
                state["skip"] = True
                print(f"  [skip] model unavailable: {model}")
                return
            except CallFailed as e:
                print(f"  [failed] {model}: {str(e)[:80]}")
                tick("fail")
                return
        rec = {**p, "model": model, "alignment_stage": stage, "temperature": gen["temperature"],
               "repeat": rep, "raw_response": text, "max_tokens": mt,
               "finish_reason": meta.get("finish_reason"), "prompt_tokens": meta.get("prompt_tokens"),
               "completion_tokens": meta.get("completion_tokens"),
               "reasoning_chars": meta.get("reasoning_chars", 0),
               "reasoning_content": meta.get("reasoning_content", ""),
               "system_prompt": system_prompt or "", "collection": collection,
               "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        if cost_writer:
            cost_writer.writerow([model, meta.get("prompt_tokens", 0), meta.get("completion_tokens", 0)])
        tick("ok")

    if already:
        print(f"     [resume] {model}: {already}/{total} already on disk")
    tasks = [one(p, rep) for p in prompts for rep in range(repeats)]
    if tasks:
        await tasks[0]
        if not state["skip"]:
            await asyncio.gather(*tasks[1:])
    fout.close()
    return state["skip"]


async def run(mock=False, smoke=False, grid_path=None, models=None, repeats=None, formats=None,
              gen_overrides=None, model_overrides=None, collection="june2026"):
    provider = get_provider(mock=mock)
    gen = dict(config.EXP["generation"])
    if gen_overrides:
        gen.update(gen_overrides)
    grid_path = grid_path or config.path("prompts")
    if models is None:
        models = config.MODELS
        if mock:
            models = [{"name": n, "stage": ("frontier" if n.endswith("-d") else "instruct")}
                      for n in (await provider.list_models())]
    filter_dims = None
    if smoke:
        sk = config.EXP["smoke"]
        models = models[:sk["models"]]
        filter_dims = set(sk["dimensions"])
        repeats = sk["repeats"]
    prompts = load_prompts(grid_path, filter_dims=filter_dims, formats=formats)
    print(f"[generate] models={len(models)} prompts={len(prompts)} mock={mock} collection={collection}")
    cost_f = open(config.path("cost_log"), "a", newline="")
    cost_writer = csv.writer(cost_f)
    used = 0
    for m in models:
        stage = m.get("stage", "instruct")
        reps = repeats if repeats is not None else m.get("repeats") or gen["repeats"].get(stage, 5)
        out_path = config.raw_dir() / (m["name"].replace("/", "_") + ".jsonl")
        override = dict((model_overrides or {}).get(m["name"], {}) or {})
        override.update({k: v for k, v in m.items() if k in ("max_tokens", "system_prompt", "extra_body")})
        print(f"  -> {m['name']} (stage={stage}, repeats={reps}, overrides={ {k: v for k, v in override.items() if k != 'extra_body'} })")
        skipped = await _run_one_model(provider, m["name"], stage, prompts, reps, out_path, cost_writer,
                                       gen, override, collection)
        used += 0 if skipped else 1
    cost_f.close()
    print(f"[generate] done. {used}/{len(models)} models used. Raw data in {config.raw_dir()}")
