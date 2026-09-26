"""Token-budget diagnosis for the June 2026 collection (Reviewer 1 #4 and #8).

In the June code every response line written to data/raw/<model>.jsonl was immediately
followed by one row (model, prompt_tokens, completion_tokens) in data/raw/_usage.csv.
Within a model, line n of the jsonl therefore corresponds to usage row n, as long as the
jsonl was never rewritten (quarantine rewrote only the files it was applied to) and no
usage rows were lost. When the counts match, every response gets its exact
completion_tokens; a rating response with completion_tokens equal to 12 or 40 was cut
off by that budget. When they do not match, only aggregate counts are reported.

Outputs (data/results/):
  U_usage_budget_hits.csv     per model: logged calls, completions at exactly 12 / 40 / 256
  U_rating_tokens_by_model.csv per model (aligned models): rating responses by budget hit,
                              largest rating completion, and the implied rating budget
  U_rating_tokens_by_category.csv per model x outcome category x budget hit
Usage:  python -m tools.diagnose_usage
"""
import json
import pandas as pd
from src import config
from src.rating_parse import classify


def _jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                out.append(None)
    return out


def main():
    rd = config.results_dir()
    up = config.path("cost_log")
    u = pd.read_csv(up, header=None, names=["model", "prompt_tokens", "completion_tokens"])
    u["completion_tokens"] = pd.to_numeric(u["completion_tokens"], errors="coerce")
    u = u[~u["model"].astype(str).str.startswith("mock/")]

    agg = []
    for m, g in u.groupby("model", sort=False):
        c = g["completion_tokens"]
        agg.append(dict(model=m, calls_logged=len(g), at_12=int((c == 12).sum()), at_40=int((c == 40).sum()),
                        at_256=int((c == 256).sum()), median=float(c.median()), max=float(c.max()),
                        excluded=m in config.excluded_models()))
    pd.DataFrame(agg).to_csv(rd / "U_usage_budget_hits.csv", index=False)

    per_model, per_cat, lines = [], [], []
    for m, g in u.groupby("model", sort=False):
        path = config.raw_dir() / (m.replace("/", "_") + ".jsonl")
        if not path.exists():
            continue
        recs = _jsonl(path)
        if len(recs) != len(g) or any(r is None for r in recs):
            lines.append(f"  {m}: NOT aligned (jsonl lines {len(recs)}, usage rows {len(g)}); aggregate counts only")
            per_model.append(dict(model=m, aligned=False, jsonl_lines=len(recs), usage_rows=len(g)))
            continue
        d = pd.DataFrame([dict(format=r.get("format"), language=r.get("language"), agent=r.get("agent"),
                               raw=r.get("raw_response") or "") for r in recs])
        d["completion_tokens"] = g["completion_tokens"].values
        rt = d[d["format"] == "rating"].copy()
        ft = d[d["format"] != "rating"]
        rt["category"] = [classify(t, l)["category"] for t, l in zip(rt["raw"], rt["language"])]
        rt["budget_hit"] = rt["completion_tokens"].map(lambda c: "at_12" if c == 12 else ("at_40" if c == 40 else "below_budget"))
        mx = rt["completion_tokens"].max()
        implied = "12" if mx <= 12 else ("40" if mx <= 40 else f">40 (max {int(mx)})")
        per_model.append(dict(model=m, aligned=True, rating_responses=len(rt),
                              rating_at_12=int((rt["completion_tokens"] == 12).sum()),
                              rating_at_40=int((rt["completion_tokens"] == 40).sum()),
                              rating_between_13_39=int(rt["completion_tokens"].between(13, 39).sum()),
                              rating_max_completion=int(mx), implied_rating_budget=implied,
                              freetext_at_256=int((ft["completion_tokens"] == 256).sum()), freetext_responses=len(ft)))
        pc = rt.groupby(["category", "budget_hit"]).size().rename("n").reset_index()
        pc.insert(0, "model", m)
        per_cat.append(pc)
        lines.append(f"  {m}: aligned; rating budget used = {implied}; {per_model[-1]['rating_at_12']} rating "
                     f"responses cut at 12, {per_model[-1]['rating_at_40']} cut at 40 (of {len(rt)})")

    pm = pd.DataFrame(per_model)
    pm.to_csv(rd / "U_rating_tokens_by_model.csv", index=False)
    if per_cat:
        pd.concat(per_cat).to_csv(rd / "U_rating_tokens_by_category.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(agg).to_string(index=False))
    print("\nAlignment of responses with usage rows:")
    print("\n".join(lines))
    print(f"\nWritten to {rd}: U_usage_budget_hits.csv, U_rating_tokens_by_model.csv, U_rating_tokens_by_category.csv")


if __name__ == "__main__":
    main()
