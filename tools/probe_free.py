"""Before the revision-round collection: which endpoint answers for each API-served model?

    python -m tools.probe_free

Reads the endpoints and each model's `serve` list from config/collection_r1.yaml and makes one
neutral call per API candidate ("Reply with the single word OK."; nothing it returns is study data).
Unlike the collection, it tries ALL candidates (for the record) and then reports which one the
collection rule would use (the first that answers, in the listed order). `local` entries (the GPU
notebook) cannot be probed from here; for models served there first, the API entries are only the
fallback. Also probes the two back-translation models.
Keys are read from .env: DASHSCOPE_API_KEY (Bailian), MODELSCOPE_API_KEY (optional),
NVIDIA_API_KEY, DEEPSEEK_API_KEY (optional). A missing key is reported, not an error.
Writes data/model_availability/free_probe_<UTC date>_<HHMM>.csv (new file per run).
"""
import asyncio
import csv
import datetime as dt
import os

from src import config
from src.providers import check_candidates, make_provider, CHECK_FIELDS, EndpointNotConfigured


def _load_cfg():
    import yaml
    with open(config.CONFIG_DIR / "collection_r1.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def main():
    cfg = _load_cfg()
    endpoints = cfg["endpoints"]
    print("Keys found in .env / environment:")
    for ep, spec in endpoints.items():
        if spec.get("api_key_env"):
            print(f"  {ep:11s} {spec['api_key_env']:20s} {'yes' if os.environ.get(spec['api_key_env']) else 'NO'}")
    # API entries only: `local` entries are served on the GPU notebook and cannot be probed from here.
    targets, local_first = [], []
    for m in cfg["anchor_study"]["models"] + cfg["rerun_excluded"]["models"]:
        if m["serve"] and m["serve"][0]["endpoint"] == "local":
            local_first.append(m["name"])
        api = [e for e in m["serve"] if e["endpoint"] != "local"]
        if api:
            targets.append(dict(m, serve=api))
    targets += [dict(name=f"translator:{t['label']}", serve=t["serve"]) for t in cfg["back_translation"]["translators"]]

    # listing (for the record): does each API list anything that looks like the study models?
    rows = []
    for ep in ("bailian", "modelscope"):
        try:
            prov = make_provider(ep, endpoints)
            ids = await prov.list_models()
            hits = [i for i in ids if any(k in i.lower() for k in ("qwen3-next", "llama", "mixtral", "gemma-2", "phi-4",
                                                                   "nemotron", "deepseek", "qwen3-max", "qwen-max"))]
            print(f"\n{ep}: GET /models lists {len(ids)} models; relevant: {', '.join(sorted(hits)[:40]) or 'none'}")
            rows.append(dict(checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                             model="(list)", endpoint=ep, host=prov.host, status=f"listed {len(ids)}",
                             error="; ".join(sorted(hits))[:2000]))
        except EndpointNotConfigured as e:
            print(f"\n{ep}: not configured ({e})")
        except Exception as e:
            print(f"\n{ep}: GET /models failed ({type(e).__name__}: {str(e)[:120]})")
            rows.append(dict(checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                             model="(list)", endpoint=ep, status="list_failed", error=str(e)[:300]))

    print("\nNeutral call per candidate:")
    rule = {}
    for m in targets:
        _, rs = await check_candidates(m, endpoints, stop_at_first=False)
        for r in rs:
            print(f"  {m['name']:40s} {r.get('endpoint', ''):11s} {str(r.get('api_model', '')):42s} "
                  f"{r.get('status', ''):22s} {str(r.get('http_status') or ''):5s} {str(r.get('error', ''))[:70]}")
            if r.get("status") == "ok" and m["name"] not in rule:
                rule[m["name"]] = f"{r['endpoint']} ({r['api_model']})"
        rows += rs
        await asyncio.sleep(0.5)

    d = config.ROOT / "data" / "model_availability"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"free_probe_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M')}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CHECK_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("\nWhat the collection rule would use (first candidate that answered):")
    for m in targets:
        if m["name"] in local_first:
            print(f"  {m['name']:44s} -> GPU notebook (local); API fallback if the notebook fails: "
                  f"{rule.get(m['name'], 'none')}")
        else:
            print(f"  {m['name']:44s} -> {rule.get(m['name'], 'NOT COLLECTABLE (no candidate answered)')}")
    others = [n for n in local_first if n not in {m['name'] for m in targets}]
    if others:
        print(f"  Served only on the GPU notebook (not probed here): {', '.join(others)}")
    print(f"Written: {out}")


if __name__ == "__main__":
    asyncio.run(main())
