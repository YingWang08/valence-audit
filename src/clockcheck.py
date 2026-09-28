"""Clock check before any timestamped collection.

Compares the machine's UTC clock with the HTTP `Date` header of well-known servers and stops if
the difference exceeds MAX_SKEW seconds. Added after a remote host recorded times about 13 hours
off (2026-09-27). The result is appended to <log_dir>/clock_checks.csv.

    python -m src.clockcheck            (prints the result; exit code 1 if the clock is off)
"""
import csv
import datetime as dt
import email.utils
import pathlib
import sys
import time
import urllib.error
import urllib.request

MAX_SKEW = 120  # seconds
URLS = ("https://www.aliyun.com", "https://www.baidu.com", "https://www.modelscope.cn",
        "https://integrate.api.nvidia.com", "https://www.cloudflare.com")


def _server_time(url, timeout=10):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "clockcheck"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            date = r.headers.get("Date")
    except urllib.error.HTTPError as e:  # an error page still carries a Date header
        date = e.headers.get("Date") if e.headers else None
    t1 = time.time()
    if not date:
        return None
    server = email.utils.parsedate_to_datetime(date).timestamp()
    return server, (t0 + t1) / 2


def check_clock(log_dir=None, fatal=True, urls=URLS):
    rows = []
    for u in urls:
        try:
            got = _server_time(u)
        except Exception as e:
            rows.append(dict(url=u, server_utc="", local_utc="", skew_s="", error=f"{type(e).__name__}"))
            continue
        if got:
            server, local = got
            rows.append(dict(url=u, server_utc=dt.datetime.fromtimestamp(server, dt.timezone.utc).isoformat(),
                             local_utc=dt.datetime.fromtimestamp(local, dt.timezone.utc).isoformat(timespec="seconds"),
                             skew_s=round(local - server, 1), error=""))
            if len([r for r in rows if r["skew_s"] != ""]) >= 2:
                break
    ok_rows = [r for r in rows if r["skew_s"] != ""]
    if log_dir:
        p = pathlib.Path(log_dir) / "clock_checks.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        new = not p.exists()
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["checked_utc", "url", "server_utc", "local_utc", "skew_s", "error"])
            if new:
                w.writeheader()
            now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            for r in rows:
                w.writerow(dict(r, checked_utc=now))
    if not ok_rows:
        msg = "[clock] could not reach any time source; clock NOT checked (logged)."
        print(msg)
        return None
    skew = sorted(r["skew_s"] for r in ok_rows)[len(ok_rows) // 2]
    print(f"[clock] local UTC minus server time: {skew:+.1f} s ({ok_rows[0]['url']})")
    if abs(skew) > MAX_SKEW:
        msg = (f"[clock] The machine clock is off by {skew:+.0f} s (> {MAX_SKEW} s). Fix the system time "
               f"(and time zone) before collecting; nothing was collected.")
        if fatal:
            raise SystemExit(msg)
        print(msg)
    return skew


if __name__ == "__main__":
    s = check_clock(fatal=False)
    sys.exit(1 if (s is not None and abs(s) > MAX_SKEW) else 0)
