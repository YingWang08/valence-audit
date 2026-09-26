"""Representative raw responses for the S1 File (Reviewer 1 #8).
  retained models: up to K responses per model x outcome category (fixed seed)
  excluded models: empty-response counts by format and collection round
                   (data/raw = final round, data/raw_quarantine = first round), and every
                   non-empty response (there are few)
Usage:  python -m tools.dump_raw_examples [K]
Writes data/results/S_raw_examples_retained.csv, S_raw_examples_excluded.csv, S_raw_examples.md
"""
import sys
import json
import glob
import pandas as pd
from src import config


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rd = config.results_dir()
    rr = pd.read_csv(config.path("rating_responses"), low_memory=False)
    ex = (rr.sample(frac=1, random_state=1).groupby(["model", "category"]).head(k)
            .sort_values(["model", "category"])
            [["model", "category", "language", "agent", "dimension", "template", "raw_response",
              "strict_value", "legacy_value"]])
    ex.to_csv(rd / "S_raw_examples_retained.csv", index=False)

    excl = set(config.excluded_models())
    rows, nonempty = [], []
    for src, d in (("final round (data/raw)", config.raw_dir()),
                   ("first round (data/raw_quarantine)", config._resolve(config.EXP["paths"]["quarantine_dir"]))):
        for p in sorted(glob.glob(str(d / "*.jsonl"))):
            for line in open(p, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("model") not in excl:
                    break
                resp = str(r.get("raw_response") or "")
                rows.append(dict(model=r["model"], round=src, format=r.get("format"), empty=not resp.strip()))
                if resp.strip():
                    nonempty.append(dict(model=r["model"], round=src, format=r.get("format"),
                                         language=r.get("language"), raw_response=resp[:600]))
    t = pd.DataFrame(rows)
    summ = t.groupby(["model", "round", "format"])["empty"].agg(["size", "sum"]).rename(columns={"size": "n", "sum": "empty"}) \
        if len(t) else pd.DataFrame()
    if len(summ):
        summ["pct_empty"] = (100 * summ["empty"] / summ["n"]).round(1)
        summ.to_csv(rd / "S_raw_examples_excluded_counts.csv")
    pd.DataFrame(nonempty).to_csv(rd / "S_raw_examples_excluded.csv", index=False)

    md = ["# Raw response examples (S1 File)", "", "## Retained models, by outcome category", ""]
    for (m, c), g in ex.groupby(["model", "category"]):
        md.append(f"**{m} / {c}**")
        for _, r in g.iterrows():
            txt = ("" if pd.isna(r["raw_response"]) else str(r["raw_response"])).replace("\n", " ")[:300]
            md.append(f"- [{r['language']}, {r['agent']}, {r['dimension']}, t{r['template']}] `{txt}`")
        md.append("")
    md += ["## Excluded models: empty responses by round and format", "", summ.to_string() if len(summ) else "(no files found)",
           "", f"Non-empty responses of excluded models: {len(nonempty)} (all listed in S_raw_examples_excluded.csv)"]
    with open(rd / "S_raw_examples.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"written to {rd} (S_raw_examples.md)")


if __name__ == "__main__":
    main()
