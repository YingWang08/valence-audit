"""Check which of the study's models the endpoint (NVIDIA NIM) still serves: one short neutral call each.

    python -m tools.probe_models

Models checked: the nine June 2026 models (seven retained, two excluded; config/models.yaml and
config/experiment.yaml) and the revision-round models (config/collection_r1.yaml), all on NIM.
Since 2026-09-27 the revision round serves the June checkpoints elsewhere (see that file and
tools/probe_free.py); this tool remains the record of what NIM itself serves. For each it records whether the model appears in the endpoint's
model list and whether a single chat call succeeds. The call uses a neutral prompt unrelated to
the study ("Reply with the single word OK."), so nothing it returns is study data. A retired
model returns HTTP 410 with its end-of-life date.
Writes data/model_availability/nim_<UTC date>_<HHMM>.csv and the full model list next to it
(a new pair of files per run; earlier records are never overwritten).
"""
import asyncio
import csv
import datetime as dt
from src import config

PROMPT = "Reply with the single word OK."


def _models():
    roles = {}

    def add(name, role):
        roles.setdefault(name, [])
        if role not in roles[name]:
            roles[name].append(role)

    for m in config.MODELS:
        add(m["name"], "june_retained")
    for n in config.excluded_models():
        add(n, "june_excluded")
    c = config.load_yaml("collection_r1.yaml")
    for m in c.get("anchor_study", {}).get("models", []):
        add(m["name"], "r1_anchor")
    for m in c.get("rerun_excluded", {}).get("models", []):
        add(m["name"], "r1_rerun")
    return [(n, ";".join(r)) for n, r in roles.items()]


async def _probe(client, model):
    try:
        r = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": PROMPT}],
            temperature=0, max_tokens=16, timeout=90)
        ch = r.choices[0]
        return dict(status="ok", http_status=200, finish_reason=getattr(ch, "finish_reason", None),
                    content_chars=len(ch.message.content or ""), error="")
    except Exception as e:  # record the error class and HTTP status as returned, no retries
        code = getattr(e, "status_code", None)
        status = "not_found" if code == 404 else ("gone" if code == 410 else "error")
        return dict(status=status, http_status=code, finish_reason=None, content_chars=None,
                    error=f"{type(e).__name__}: {str(e)[:300]}")


async def main():
    from src.providers import NimProvider
    p = NimProvider()
    listed = sorted(m.id for m in (await p.client.models.list()).data)
    now = dt.datetime.now(dt.timezone.utc)
    rows = []
    for name, role in _models():
        res = await _probe(p.client, name)
        rows.append(dict(checked_utc=now.isoformat(timespec="seconds"), model=name, role=role,
                         in_model_list=name in listed, **res))
        await asyncio.sleep(1)
    d = config.data_root() / "model_availability"
    d.mkdir(parents=True, exist_ok=True)
    tag = now.strftime("%Y%m%d_%H%M")
    out_csv, out_txt = d / f"nim_{tag}.csv", d / f"nim_models_list_{tag}.txt"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out_txt.write_text(f"# GET {config.EXP['api']['base_url']}/models at {now.isoformat(timespec='seconds')}\n"
                       + "\n".join(listed) + "\n", encoding="utf-8")
    print(f"\n{'model':45s} {'role':24s} {'listed':6s} {'status':10s} http  detail")
    for r in rows:
        det = r["error"][:60] if r["error"] else f"finish={r['finish_reason']} chars={r['content_chars']}"
        print(f"{r['model']:45s} {r['role']:24s} {str(r['in_model_list']):6s} {r['status']:10s} "
              f"{str(r['http_status']):4s}  {det}")
    print(f"\nWritten: {out_csv} and {out_txt.name}")


if __name__ == "__main__":
    asyncio.run(main())
