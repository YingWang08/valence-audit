"""Validation of automated coding against human coding (Reviewer 1 #4 and #11).

  python -m tools.validation export [N_RATING] [N_FREE]
      writes data/validation/rating_sample.csv and freetext_sample.csv with EMPTY coding
      columns and a coding guide. Automated codes are kept in a separate key file so that
      coders work blind to them.
  python -m tools.validation score
      reads the coded files and writes data/validation/validation_results.csv and .md:
        rating: agreement of strict and legacy parsers with the human-coded value
                (exact value; "is a usable rating" with Cohen's kappa)
        free text: precision, recall and kappa of the hedge and refusal lexicon flags
        two coders: inter-coder kappa when columns *_2 are filled in

Sampling (fixed seed): the rating sample over-represents responses where the two parsers
disagree and responses the strict parser rejects, because those determine whether the
parser change matters; the free-text sample is stratified by the lexicon flags.
Sampling weights are stored so that population-level accuracy can be re-weighted.
"""
import sys
import numpy as np
import pandas as pd
from src import config

GUIDE = """Coding guide
============
rating_sample.csv (one prompt asked a model to rate ONE referent on a 1-7 scale):
  human_value     the rating the response gives, as a number (e.g. 5, or 5.5 for "5-6").
                  Leave EMPTY if the response gives no rating.
  human_category  one of: rating | refusal | hedge | truncated | out_of_range | other
                  (rating = a usable 1-7 rating was given, even if hedged)
freetext_sample.csv (free-text comparison of humans and AI systems):
  human_hedge     1 if the response avoids choosing / says it depends / both have merits, else 0
  human_refuse    1 if the response declines to answer or disclaims ability to judge, else 0
Second coder: fill human_value_2 / human_category_2 / human_hedge_2 / human_refuse_2.
Do not open *_key.csv until coding is finished.
"""


def _vdir():
    d = config.data_root() / "validation"
    d.mkdir(parents=True, exist_ok=True)
    return d


def export(n_rating=300, n_free=200, seed=7):
    rr = pd.read_csv(config.path("rating_responses"), low_memory=False)
    rr = rr[~rr["model"].isin(config.excluded_models())].reset_index(drop=True)
    valid = rr["category"].isin(["valid", "valid_range"])
    disagree = (rr["legacy_value"].fillna(-1) != rr["strict_value"].fillna(-1))
    strata = {"parsers_disagree": disagree, "strict_invalid_agree": ~valid & ~disagree, "strict_valid_agree": valid & ~disagree}
    per = n_rating // 3
    parts = []
    for name, mask in strata.items():
        pool = rr[mask]
        take = pool.sample(min(per, len(pool)), random_state=seed)
        take = take.assign(stratum=name, weight=len(pool) / max(len(take), 1))
        parts.append(take)
    s = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    s["id"] = [f"R{i:04d}" for i in range(len(s))]
    s[["id", "model", "language", "agent", "dimension", "template", "raw_response"]] \
        .assign(human_value="", human_category="", human_value_2="", human_category_2="") \
        .to_csv(_vdir() / "rating_sample.csv", index=False, encoding="utf-8-sig")
    s[["id", "stratum", "weight", "category", "strict_value", "legacy_value"]].to_csv(_vdir() / "rating_sample_key.csv", index=False)

    fr = pd.read_csv(config.path("freetext_responses"), low_memory=False)
    fr = fr[~fr["model"].isin(config.excluded_models()) & (fr["empty"] == 0)].reset_index(drop=True)
    fr["flag"] = fr["hedge"].astype(int).astype(str) + fr["refuse"].astype(int).astype(str)
    parts = []
    for flag, g in fr.groupby("flag"):
        take = g.sample(min(n_free // 4, len(g)), random_state=seed)
        parts.append(take.assign(weight=len(g) / max(len(take), 1)))
    f = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    f["id"] = [f"F{i:04d}" for i in range(len(f))]
    f[["id", "model", "language", "format", "dimension", "raw_response"]] \
        .assign(human_hedge="", human_refuse="", human_hedge_2="", human_refuse_2="") \
        .to_csv(_vdir() / "freetext_sample.csv", index=False, encoding="utf-8-sig")
    f[["id", "flag", "weight", "hedge", "refuse"]].to_csv(_vdir() / "freetext_sample_key.csv", index=False)
    with open(_vdir() / "CODING_GUIDE.txt", "w", encoding="utf-8") as fh:
        fh.write(GUIDE)
    print(f"exported {len(s)} rating and {len(f)} free-text responses to {_vdir()}")


def _read(path):
    """Coder files may come back from Excel as UTF-8 (with or without BOM) or as GBK."""
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"cannot read {path}: save it from Excel as 'CSV UTF-8 (comma delimited)'")


def kappa(a, b):
    a, b = np.asarray(a).astype(str), np.asarray(b).astype(str)
    if len(a) == 0:
        return np.nan
    cats = sorted(set(a) | set(b))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def _prf(pred, truth, w):
    pred, truth, w = np.asarray(pred, int), np.asarray(truth, int), np.asarray(w, float)
    tp = (w * (pred & truth)).sum()
    fp = (w * (pred & ~truth.astype(bool))).sum()
    fn = (w * (~pred.astype(bool) & truth)).sum()
    return (tp / (tp + fp) if tp + fp else np.nan), (tp / (tp + fn) if tp + fn else np.nan)


def score():
    out = []
    r = _read(_vdir() / "rating_sample.csv").merge(pd.read_csv(_vdir() / "rating_sample_key.csv"), on="id")
    r = r[r["human_category"].notna() & (r["human_category"].astype(str).str.strip() != "")]
    if len(r):
        hv = pd.to_numeric(r["human_value"], errors="coerce")
        usable = hv.notna()
        for parser, col in (("strict", "strict_value"), ("legacy", "legacy_value")):
            pv = pd.to_numeric(r[col], errors="coerce")
            same = (pv.isna() & hv.isna()) | (np.isclose(pv.fillna(-9), hv.fillna(-8)))
            w = r["weight"]
            out.append(dict(check=f"rating value agreement, {parser} parser", n=len(r),
                            unweighted=same.mean(), weighted=(w * same).sum() / w.sum(),
                            kappa_usable=kappa(pv.notna(), usable),
                            false_numbers=int((pv.notna() & hv.isna()).sum()),
                            missed_numbers=int((pv.isna() & hv.notna()).sum())))
        if r.get("human_category_2") is not None and r["human_category_2"].notna().any():
            both = r[r["human_category_2"].notna()]
            out.append(dict(check="inter-coder kappa, rating category", n=len(both),
                            kappa_usable=kappa(both["human_category"], both["human_category_2"])))
    f = _read(_vdir() / "freetext_sample.csv").merge(pd.read_csv(_vdir() / "freetext_sample_key.csv"), on="id")
    f = f[pd.to_numeric(f["human_hedge"], errors="coerce").notna()]
    if len(f):
        for flag in ("hedge", "refuse"):
            truth = pd.to_numeric(f[f"human_{flag}"], errors="coerce").astype(int)
            prec, rec = _prf(f[flag].astype(int), truth, f["weight"])
            out.append(dict(check=f"free-text {flag} lexicon", n=len(f), unweighted=(f[flag].astype(int) == truth).mean(),
                            precision_weighted=prec, recall_weighted=rec, kappa_usable=kappa(f[flag].astype(int), truth)))
            c2 = f"human_{flag}_2"
            if c2 in f and pd.to_numeric(f[c2], errors="coerce").notna().any():
                both = f[pd.to_numeric(f[c2], errors="coerce").notna()]
                out.append(dict(check=f"inter-coder kappa, {flag}", n=len(both),
                                kappa_usable=kappa(both[f"human_{flag}"].astype(int), pd.to_numeric(both[c2]).astype(int))))
    t = pd.DataFrame(out)
    t.to_csv(_vdir() / "validation_results.csv", index=False)
    with open(_vdir() / "validation_results.md", "w", encoding="utf-8") as fh:
        fh.write(t.to_string(index=False))
    print(t.to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "export":
        export(*(int(x) for x in sys.argv[2:4]))
    elif cmd == "score":
        score()
    else:
        print(__doc__)
