"""Find which chat models the endpoint (NVIDIA NIM) actually serves: one short neutral call each.

    python -m tools.scan_models

Why this exists: the 2026-09-27 probe (tools/probe_models.py) showed that appearing in
GET /models does not mean a model can be called. Four listed panel models returned HTTP 404 on a
chat call. Replacement models for the revision-round panel are therefore chosen only from models
that answer a call, and this scan is the record of that.

Scope: every listed model from the developers of the June 2026 models (Meta, Mistral, Qwen,
Google, Microsoft, NVIDIA) plus openai/ (gpt-oss-20b re-collection). Names that are clearly not
chat models (embedding, reranking, guard, reward, parsing) are listed in the output but not called.
The call uses the same neutral prompt as tools/probe_models.py ("Reply with the single word OK."),
so nothing it returns is study data. No SDK retries; a model that times out, cannot be reached,
is rate-limited (429) or hits a server error (5xx) is called once more at the end, and both
outcomes are recorded (first_status, status).
Writes data/model_availability/nim_scan_<UTC date>_<HHMM>.csv (a new file per run; earlier
records are never overwritten).
"""
import asyncio
import csv
import datetime as dt
import time

from src import config

PROMPT = "Reply with the single word OK."
DEVELOPERS = ("meta/", "mistralai/", "qwen/", "google/", "microsoft/", "nvidia/", "openai/")
NON_CHAT = ("embed", "rerank", "retriever", "guard", "safety", "reward", "parse", "ocr")
TIMEOUT = 45   # seconds per call
PAUSE = 1.5    # seconds between calls (free-tier rate limit)
FIELDS = ["checked_utc", "model", "status", "http_status", "seconds", "finish_reason",
          "content_chars", "first_status", "attempts", "error"]


def _utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


async def _call(client, model):
    from openai import APIConnectionError, APITimeoutError
    t0 = time.monotonic()
    res = dict(checked_utc=_utc(), model=model, finish_reason=None, content_chars=None, error="")
    try:
        r = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": PROMPT}],
            temperature=0, max_tokens=16)
        ch = r.choices[0]
        res.update(status="ok", http_status=200, finish_reason=getattr(ch, "finish_reason", None),
                   content_chars=len(ch.message.content or ""))
    except APIConnectionError as e:  # no HTTP response (APITimeoutError is a subclass)
        res.update(status="timeout" if isinstance(e, APITimeoutError) else "connection_error",
                   http_status=None, error=f"{type(e).__name__}: {str(e)[:500]}")
    except Exception as e:
        code = getattr(e, "status_code", None)
        res.update(status={404: "not_found", 410: "gone"}.get(code, "error"), http_status=code,
                   error=f"{type(e).__name__}: {str(e)[:500]}")
    res["seconds"] = round(time.monotonic() - t0, 1)
    return res


async def main():
    from src.providers import NimProvider
    base = NimProvider().client
    lister = base.with_options(timeout=30, max_retries=4)
    caller = base.with_options(timeout=TIMEOUT, max_retries=0)

    listed = sorted(m.id for m in (await lister.models.list()).data)
    cands, skipped = [], []
    for m in listed:
        low = m.lower()
        if low.startswith(DEVELOPERS):
            (skipped if any(k in low for k in NON_CHAT) else cands).append(m)
    print(f"{len(listed)} models listed; {len(cands)} to call, {len(skipped)} not chat models.\n")

    results = {}
    for i, m in enumerate(cands, 1):
        r = await _call(caller, m)
        r.update(first_status=r["status"], attempts=1)
        results[m] = r
        print(f"[{i:3d}/{len(cands)}] {m:55s} {r['status']:16s} {r['seconds']}s")
        await asyncio.sleep(PAUSE)

    again = [m for m, r in results.items() if r["status"] in ("timeout", "connection_error")
             or (r["http_status"] or 0) == 429 or (r["http_status"] or 0) >= 500]
    if again:
        print(f"\nCalling {len(again)} model(s) that timed out, were unreachable, "
              f"rate-limited or hit a server error once more ...")
    for m in again:
        r = await _call(caller, m)
        r.update(first_status=results[m]["status"], attempts=2)
        results[m] = r
        print(f"  {m:55s} {r['status']:16s} {r['seconds']}s")
        await asyncio.sleep(PAUSE)

    rows = [results[m] for m in cands]
    rows += [dict(checked_utc=_utc(), model=m, status="not_called_non_chat", attempts=0)
             for m in skipped]
    d = config.data_root() / "model_availability"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"nim_scan_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M')}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print("\nCallable (HTTP 200):")
    for r in rows:
        if r["status"] == "ok":
            print(f"  {r['model']}")
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    asyncio.run(main())
