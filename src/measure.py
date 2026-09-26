"""Measurement: raw responses -> per-response tables and cell-level asymmetries.

Outputs (in <data root>/measured/):
  rating_responses.csv  one row per rating-format response of every RETAINED model, with the
                        strict value, the legacy (v1.0.0) value, the outcome category and
                        flags (empty / refusal / hedge / truncated / out-of-range / malformed).
                        This is the item-level file requested by Reviewers 1 and 3.
  freetext_responses.csv one row per free-text response (hedge / refuse flags, text_m1).
  items.parquet / items.csv  long format used by analyze_lme4.R:
                        measure = rating         cell asymmetry, strict parser (primary)
                                  rating_legacy  cell asymmetry, v1.0.0 parser (comparison only)
                                  text_m1 / hedge / refuse  (exploratory)
                        Cell = model x dimension x language x template;
                        a = [mean(AI) - mean(human)] / 6.
  excluded_models_summary.csv  empty-response counts of the excluded models by format,
                        for data/raw and data/raw_quarantine separately.
  measure_log.json      counters.

Excluded models (config/experiment.yaml: analysis.exclude_models) never enter the tables
above (v1.0.0 let one row through and mixed their rows into items.csv).
"""
import os
import json
import glob
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from src import config
from src.rating_parse import classify, VALID_CATEGORIES


def _iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue


def _raw_files(raw_dir):
    out = []
    for p in sorted(glob.glob(str(raw_dir / "*.jsonl"))):
        base = os.path.basename(p)
        if base.startswith("_"):
            continue
        out.append(p)
    return out


def summarize_excluded(excl, raw_dir, qdir):
    rows = []
    for source, d in (("raw", raw_dir), ("raw_quarantine", qdir)):
        if not d.exists():
            continue
        for p in sorted(glob.glob(str(d / "*.jsonl"))):
            cnt = defaultdict(lambda: [0, 0])
            model = None
            for r in _iter_jsonl(p):
                model = r.get("model", model)
                if model not in excl:
                    break
                c = cnt[r.get("format", "?")]
                c[0] += 1
                c[1] += int(not str(r.get("raw_response") or "").strip())
            if model in excl:
                for fmt, (n, e) in sorted(cnt.items()):
                    rows.append(dict(model=model, source=source, format=fmt, n_responses=n, n_empty=e,
                                     pct_empty=round(100 * e / n, 1) if n else np.nan))
    return pd.DataFrame(rows)


def measure(verbose=True):
    mc = config.EXP["measurement"]
    th = mc.get("truncation_heuristic", {}) or {}
    comp_on = bool(mc.get("comparative_attribution", False))
    comp_mag = float(mc.get("comparative_magnitude", 0.5))
    use_m1 = mc.get("enable_m1", True)
    excl = set(config.excluded_models())
    raw_dir = config.raw_dir()

    ft = None
    if use_m1:
        from src import freetext as ft

    rating_rows, free_rows, item_rows = [], [], []
    counters = Counter()
    skipped_files = []
    for path in _raw_files(raw_dir):
        for r in _iter_jsonl(path):
            model = r.get("model")
            if model in excl:
                skipped_files.append(os.path.basename(path))
                break
            lang = r.get("language", "en")
            fmt = r.get("format", "")
            meta = dict(model=model, family=config.family(model), alignment_stage=r.get("alignment_stage", ""),
                        dimension=r.get("dimension", ""), language=lang, format=fmt,
                        template=int(r.get("template", 0)), repeat=int(r.get("repeat", 0)),
                        prompt_id=r.get("prompt_id", ""))
            raw = r.get("raw_response", "") or ""
            if fmt == "rating":
                c = classify(raw, lang, th.get("en_min_words", 6), th.get("zh_min_chars", 12),
                             finish_reason=r.get("finish_reason"))
                rating_rows.append({**meta, "agent": r.get("agent", ""), **c,
                                    "n_chars": len(raw), "max_tokens": r.get("max_tokens"),
                                    "finish_reason": r.get("finish_reason"),
                                    "completion_tokens": r.get("completion_tokens"),
                                    "raw_response": raw})
                counters["rating_" + c["category"]] += 1
            else:
                h = r_ = np.nan
                a1 = None
                if ft is not None:
                    h = ft._flag(raw, lang, ft.HEDGE)
                    r_ = ft._flag(raw, lang, ft.REFUSE)
                    a1 = ft._attr_diff(raw, lang, ft._valence, comp_on=comp_on, comp_mag=comp_mag,
                                       counters=counters)
                free_rows.append({**meta, "agent": "both", "self_referential": fmt == "reflect",
                                  "empty": int(not raw.strip()), "hedge": h, "refuse": r_,
                                  "text_m1": a1 if a1 is not None else np.nan, "raw_response": raw})
                base = {**meta, "self_referential": fmt == "reflect"}
                if a1 is not None:
                    item_rows.append({**base, "measure": "text_m1", "asymmetry": a1})
                item_rows.append({**base, "measure": "hedge", "asymmetry": h})
                item_rows.append({**base, "measure": "refuse", "asymmetry": r_})

    rr = pd.DataFrame(rating_rows)
    fr = pd.DataFrame(free_rows)
    if rr.empty:
        raise SystemExit(f"[measure] no rating responses found in {raw_dir}.")

    # ---- cell-level asymmetries, strict and legacy ----
    key = ["model", "family", "alignment_stage", "dimension", "language", "template"]
    for measure_name, col, ok in (("rating", "strict_value", rr["category"].isin(VALID_CATEGORIES)),
                                  ("rating_legacy", "legacy_value", rr["legacy_value"].notna())):
        v = rr[ok & rr["agent"].isin(["ai", "human"])]
        g = v.groupby(key + ["agent"])[col].agg(["mean", "size"]).unstack("agent")
        g = g.dropna(subset=[("mean", "ai"), ("mean", "human")])
        for idx, row in g.iterrows():
            d = dict(zip(key, idx))
            item_rows.append({**d, "format": "rating", "repeat": -1, "prompt_id": "",
                              "self_referential": False, "measure": measure_name,
                              "asymmetry": (row[("mean", "ai")] - row[("mean", "human")]) / 6.0,
                              "mean_ai": row[("mean", "ai")], "mean_human": row[("mean", "human")],
                              "n_ai": int(row[("size", "ai")]), "n_human": int(row[("size", "human")])})
        counters[f"cells_{measure_name}"] = len(g)

    items = pd.DataFrame(item_rows)
    items["item"] = items["dimension"].astype(str) + "|" + items["format"].astype(str) + "|" + items["template"].astype(str)
    items["frame"] = items["format"].astype(str) + "|" + items["language"].astype(str) + "|" + items["template"].astype(str)

    out = config.path("measured")
    items.to_parquet(out)
    items.to_csv(str(out).replace(".parquet", ".csv"), index=False)
    rr.to_csv(config.path("rating_responses"), index=False)
    fr.to_csv(config.path("freetext_responses"), index=False)
    exs = summarize_excluded(excl, raw_dir, config._resolve(config.EXP["paths"]["quarantine_dir"]))
    exs.to_csv(out.parent / "excluded_models_summary.csv", index=False)
    log = {"retained_models": sorted(rr["model"].unique().tolist()),
           "excluded_models_skipped": sorted(set(skipped_files)),
           "n_rating_responses": int(len(rr)), "n_freetext_responses": int(len(fr)),
           **{k: int(v) for k, v in counters.items()}}
    with open(out.parent / "measure_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    if verbose:
        n = len(rr)
        cats = rr["category"].value_counts()
        print(f"\n[measure] retained models: {rr['model'].nunique()}  rating responses: {n}  free-text responses: {len(fr)}")
        print("[measure] rating outcome categories (strict parser):")
        for c_, k in cats.items():
            print(f"            {c_:13s} {k:6d}  ({100 * k / n:.1f}%)")
        fp = rr[rr["legacy_value"].notna() & ~rr["category"].isin(VALID_CATEGORIES)]
        print(f"[measure] legacy parser returned a number for {len(fp)} responses that the strict parser "
              f"classifies as non-ratings (refusal/hedge/truncated/...); see rating_responses.csv")
        print(f"[measure] complete cells: strict {counters['cells_rating']}, legacy {counters['cells_rating_legacy']}")
        if skipped_files:
            print(f"[measure] excluded-model files skipped: {sorted(set(skipped_files))}")
        print(f"[measure] wrote {out.parent}")
    return items


if __name__ == "__main__":
    measure()
