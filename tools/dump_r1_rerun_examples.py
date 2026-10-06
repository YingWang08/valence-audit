"""Representative raw responses of the two excluded models re-collected in the revision round
with a 4096-token budget (Reviewer 1 #8), for the S1 File.

For each re-collected model: up to K responses per outcome category (fixed seed), with the prompt,
the final answer, the length of the separately returned reasoning, the first characters of that
reasoning, finish_reason and token count.
Usage:  python -m tools.dump_r1_rerun_examples [K]
Reads data/r1/raw/*.jsonl (records whose collection ends in "_rerun").
Writes data/r1/results/S_r1_rerun_examples.csv and S_r1_rerun_examples.md
"""
import sys
import json
import glob
import pandas as pd
from src import config
from src.rating_parse import classify

REASONING_CHARS_SHOWN = 300


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    r1 = config.ROOT / "data" / "r1"
    rows = []
    for p in sorted(glob.glob(str(r1 / "raw" / "*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not str(r.get("collection", "")).endswith("_rerun") or r.get("format") != "rating":
                continue
            c = classify(r.get("raw_response"), r.get("language", "en"), finish_reason=r.get("finish_reason"))
            reasoning = str(r.get("reasoning_content") or "")
            rows.append(dict(model=r["model"], category=c["category"], strict_value=c["strict_value"],
                             language=r.get("language"), agent=r.get("agent"), dimension=r.get("dimension"),
                             template=r.get("template"), repeat=r.get("repeat"), prompt=r.get("text", ""),
                             raw_response=str(r.get("raw_response") or ""), reasoning_chars=len(reasoning),
                             reasoning_start=reasoning[:REASONING_CHARS_SHOWN].replace("\n", " "),
                             finish_reason=r.get("finish_reason"), completion_tokens=r.get("completion_tokens"),
                             max_tokens=r.get("max_tokens")))
    if not rows:
        raise SystemExit(f"no re-collected (\"_rerun\") rating records in {r1 / 'raw'}")
    d = pd.DataFrame(rows)
    ex = d.sample(frac=1, random_state=1).groupby(["model", "category"]).head(k).sort_values(["model", "category"])
    out = r1 / "results"
    out.mkdir(parents=True, exist_ok=True)
    ex.to_csv(out / "S_r1_rerun_examples.csv", index=False)

    counts = d.groupby(["model", "category"]).size().unstack(fill_value=0)
    md = ["# Re-collected excluded models: raw response examples (S1 File)", "",
          f"Rating prompts of June, max_tokens {int(d['max_tokens'].max())}; reasoning returned separately. "
          f"Up to {k} examples per model and outcome category (seed 1).", "", "## Outcome counts", "",
          counts.to_string(), ""]
    for (m, c), g in ex.groupby(["model", "category"]):
        md.append(f"## {m} / {c}")
        for _, r in g.iterrows():
            md.append(f"- [{r['language']}, {r['agent']}, {r['dimension']}, t{r['template']}; finish {r['finish_reason']}, "
                      f"{r['completion_tokens']} tokens, reasoning {r['reasoning_chars']} chars]")
            md.append(f"  - prompt: `{str(r['prompt']).replace(chr(10), ' ')}`")
            md.append(f"  - answer: `{r['raw_response'].replace(chr(10), ' ')[:300]}`")
            if r["reasoning_chars"]:
                md.append(f"  - reasoning (first {REASONING_CHARS_SHOWN} characters): `{r['reasoning_start']}`")
        md.append("")
    with open(out / "S_r1_rerun_examples.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"{len(ex)} examples from {d['model'].nunique()} models -> {out / 'S_r1_rerun_examples.md'}")


if __name__ == "__main__":
    main()
