"""Blinded single-coder validation of the rating parser, with test-retest (intra-rater) reliability.

The validated quantity is low-inference: which 1-7 number, if any, a rating response gives
as its answer, and which outcome category the response belongs to. Codebook:
tools/coding_manual_zh.md. Commands (run from the repository root):

  python -m tools.validation export
      Draws the sample (fixed seed), writes the blank coding workbook and practice workbook
      to data/validation/coder_A/, the automated codes to data/validation/key_DO_NOT_SHARE/,
      and data/validation/PROTOCOL.md (sample, SHA-256 hashes, analysis plan, decision rules)
      to be registered (e.g. on OSF) BEFORE coding starts.
  python -m tools.validation export-recode
      At least 7 days after coding: a blank workbook with a pre-drawn 30% subset in a new
      order (data/validation/coder_A/rating_recode_A.xlsx).
  python -m tools.validation score
      data/validation/validation_results.md / .csv, confusion matrix, the table of all
      parser disagreements with the coder's judgement (S1), and the full unblinded coded
      sample for the deposit.

Sample: every response on which the strict and legacy parsers disagree; a random sample of
responses the strict parser rejects; a random sample of responses the strict parser accepts
whose text is more than a bare number. Bare-number responses (the text is a single digit
1-7, read identically by both parsers) are verified mechanically on the whole population.
Weights (stratum size / sample size) give population-level accuracy.
An optional second coder's workbook (coder_B/rating_coder_B.xlsx) is scored if present.
"""
import sys
import json
import hashlib
import datetime as dt
import numpy as np
import pandas as pd
from src import config

SEED = 7
RECODE_FRACTION = 0.30
CATS = ["rating", "out_of_range", "refusal", "hedge", "truncated", "other"]
MAP_STRICT = {"valid": "rating", "valid_range": "rating", "out_of_range": "out_of_range", "refusal": "refusal",
              "hedge": "hedge", "truncated": "truncated", "empty": "other", "multiple": "other", "malformed": "other"}
CODE_COLS = ["code_value", "code_category"]
VALIDATIONS = {"code_value": "decimal", "code_category": '"' + ",".join(CATS) + '"'}
LEGEND = [("response", "模型的回答（原文，勿修改）"),
          ("code_value", "回答作为答案给出的分数；没有就留空。范围取中点（5-6 记 5.5）；1-7 以外的数字照写"),
          ("code_category", "rating / out_of_range / refusal / hedge / truncated / other，按手册顺序选第一个符合的"),
          ("notes", "可选：拿不准的理由")]
EXAMPLES = [("response", "code_value", "code_category"),
            ("I'd rate a human a 6 on this", 6, "rating"),
            ("As an AI, I don't have personal opinions, but", "", "truncated"),
            ("I cannot assign a moral score to a person because", "", "refusal")]


def _vdir():
    d = config.data_root() / "validation"
    d.mkdir(parents=True, exist_ok=True)
    return d


TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".py", ".r", ".txt", ".yaml", ".yml"}


def _sha(path):
    """SHA-256 of a file. Text files are hashed with CRLF normalized to LF, i.e. in the form git
    stores them (and GitHub/Zenodo serve them), so the hash is the same on Windows (where git
    checks text files out with CRLF) and on Linux/macOS, and can be checked with `sha256sum` on a
    downloaded copy. Binary files (e.g. .xlsx) are hashed byte for byte."""
    b = open(path, "rb").read()
    if str(path).lower().endswith(tuple(TEXT_SUFFIXES)):
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


# ----------------------------------------------------------------------------- workbook
def _write_book(path, df):
    from openpyxl import Workbook
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "编码"
    cols = ["id", "language", "response"] + CODE_COLS + ["notes"]
    font, bold = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
    yellow, head_yellow = PatternFill("solid", start_color="FFF2CC"), PatternFill("solid", start_color="FFD966")
    for j, c in enumerate(cols, 1):
        cell = ws.cell(row=1, column=j, value=c)
        cell.font = bold
        if c in CODE_COLS or c == "notes":
            cell.fill = head_yellow
    for i, (_, r) in enumerate(df.iterrows(), 2):
        for j, c in enumerate(cols, 1):
            if c in ("id", "language", "response"):
                v = "" if pd.isna(r[c]) else ILLEGAL_CHARACTERS_RE.sub("", str(r[c]))
                cell = ws.cell(row=i, column=j, value=v)
                cell.number_format = "@"          # keeps "5-6" and "6/7" as text (Excel would make dates)
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            else:
                cell = ws.cell(row=i, column=j)
                cell.fill = yellow
            cell.font = font
    n = len(df) + 1
    for c, formula in VALIDATIONS.items():
        L = ws.cell(row=1, column=cols.index(c) + 1).column_letter
        if formula == "decimal":
            dv = DataValidation(type="decimal", operator="between", formula1="0", formula2="100", allow_blank=True)
            dv.error, dv.errorTitle = "请输入数字，或留空", "输入无效"
        else:
            dv = DataValidation(type="list", formula1=formula, allow_blank=True)
            dv.error, dv.errorTitle = "请从下拉列表中选择", "输入无效"
        dv.showErrorMessage = True
        ws.add_data_validation(dv)
        dv.add(f"{L}2:{L}{n}")
    for c, w in {"id": 9, "language": 9, "response": 75, "code_value": 11, "code_category": 15, "notes": 30}.items():
        ws.column_dimensions[ws.cell(row=1, column=cols.index(c) + 1).column_letter].width = w
    ws.freeze_panes = "A2"
    info = wb.create_sheet("说明")
    info.column_dimensions["A"].width = 18
    info.column_dimensions["B"].width = 80
    info.cell(row=1, column=1, value="只在黄色列填写；规则见《人工编码手册》").font = bold
    r = 3
    for k, v in LEGEND:
        info.cell(row=r, column=1, value=k).font = bold
        info.cell(row=r, column=2, value=v).font = font
        r += 1
    r += 1
    info.cell(row=r, column=1, value="示例（虚构）").font = bold
    r += 1
    for ex in EXAMPLES:
        for j, v in enumerate(ex, 1):
            info.cell(row=r, column=j, value=v).font = font
        r += 1
    wb.save(path)


def _read_book(path):
    if not path.exists():
        return None
    d = pd.read_excel(path, sheet_name="编码", dtype={"id": str})
    d["code_value"] = pd.to_numeric(d["code_value"], errors="coerce")
    d["code_category"] = d["code_category"].astype("string").str.strip().str.lower()
    return d[["id", "code_value", "code_category"]]


# ------------------------------------------------------------------------------- export
def _load_rr():
    rr = pd.read_csv(config.path("rating_responses"), low_memory=False, keep_default_na=False, na_values=[""])
    rr = rr[~rr["model"].isin(config.excluded_models())].reset_index(drop=True)
    rr["raw_response"] = rr["raw_response"].fillna("")
    return rr


def _strata(rr):
    valid = rr["category"].isin(["valid", "valid_range"])
    disagree = rr["legacy_value"].fillna(-1) != rr["strict_value"].fillna(-1)
    bare = valid & ~disagree & (rr["strict_how"] == "bare")
    return {"parsers_disagree": disagree,
            "strict_invalid": ~valid & ~disagree,
            "strict_valid_not_bare": valid & ~disagree & ~bare,
            "bare_number": bare}


def export(n_invalid=100, n_valid_not_bare=100, n_practice=20):
    v = _vdir()
    existing = [p for p in (v / "PROTOCOL.md", v / "coder_A" / "rating_coder_A.xlsx") if p.exists()]
    if existing and "--force" not in sys.argv:
        sys.exit("export has already been run; re-running would overwrite the coding workbook and the registered\n"
                 "protocol. Existing: " + ", ".join(str(p) for p in existing) + "\n"
                 "Use --force only if coding has NOT started and the protocol has NOT been registered.")
    rr = _load_rr()
    st = _strata(rr)
    take_n = {"parsers_disagree": None, "strict_invalid": n_invalid, "strict_valid_not_bare": n_valid_not_bare}
    parts = []
    for name, k in take_n.items():
        pool = rr[st[name]]
        k = len(pool) if k is None else min(k, len(pool))
        s = pool.sample(k, random_state=SEED)
        parts.append(s.assign(stratum=name, pool_size=len(pool), weight=len(pool) / max(k, 1)))
    s = pd.concat(parts).sample(frac=1, random_state=SEED)
    rest = rr.drop(index=s.index)
    rest = rest[~st["bare_number"].loc[rest.index]]
    prac = rest.sample(min(n_practice, len(rest)), random_state=SEED + 1)
    s = s.reset_index(drop=True)
    s["id"] = [f"R{i:04d}" for i in range(len(s))]
    rec = s.groupby("stratum", group_keys=False).sample(frac=RECODE_FRACTION, random_state=SEED + 2)
    s["recode"] = s["id"].isin(rec["id"])
    prac = prac.reset_index(drop=True)
    prac["id"] = [f"P{i:02d}" for i in range(len(prac))]
    s["response"], prac["response"] = s["raw_response"], prac["raw_response"]

    keydir = v / "key_DO_NOT_SHARE"
    keydir.mkdir(exist_ok=True)
    s[["id", "model", "language", "agent", "dimension", "template", "repeat", "stratum", "pool_size", "weight",
       "recode", "category", "strict_value", "legacy_value", "strict_how", "raw_response"]].to_csv(keydir / "key_rating.csv", index=False)
    cd = v / "coder_A"
    cd.mkdir(exist_ok=True)
    _write_book(cd / "rating_coder_A.xlsx", s)
    _write_book(cd / "practice_rating_A.xlsx", prac)
    manual = config.ROOT / "tools" / "coding_manual_zh.md"
    (cd / "人工编码手册.md").write_text(manual.read_text(encoding="utf-8"), encoding="utf-8")

    pop = {k: int(m.sum()) for k, m in st.items()}
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.ROOT,
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    info = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), seed=SEED, git_commit=commit,
                population_rating_responses=len(rr), population_strata=pop,
                sample_strata={k: int((s["stratum"] == k).sum()) for k in take_n}, n_sample=len(s),
                n_recode=int(s["recode"].sum()), n_practice=len(prac),
                sha256={"rating_coder_A.xlsx (blank)": _sha(cd / "rating_coder_A.xlsx"),
                        "tools/coding_manual_zh.md": _sha(manual),
                        "tools/validation.py": _sha(config.ROOT / "tools" / "validation.py"),
                        "src/rating_parse.py": _sha(config.ROOT / "src" / "rating_parse.py"),
                        "rating_responses.csv": _sha(config.path("rating_responses"))})
    with open(keydir / "sample_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    _protocol(v, info)
    print(json.dumps({k: info[k] for k in ("population_strata", "sample_strata", "n_recode", "n_practice")}, indent=2))
    print(f"\n1. Register {v / 'PROTOCOL.md'} (e.g. OSF) before opening the coding workbook.\n"
          f"2. Code {cd / 'rating_coder_A.xlsx'}. Do not open {keydir.name}/.")


def _protocol(v, info):
    h = "\n".join(f"| {k} | `{x}` |" for k, x in info["sha256"].items())
    ps, ss = info["population_strata"], info["sample_strata"]
    text = f"""# Validation protocol: rating-response parser (registered before coding)

Created (UTC): {info['created_utc']}

## Purpose
Estimate how accurately the automated parsers (strict parser used in the revision; legacy parser
of version 1.0.0) read the answer of each rating-format response, and how accurately the strict
parser assigns the outcome category. The coded quantity is low-inference: which number from 1 to 7,
if any, the response gives as its answer, and which of six outcome categories applies.

## Sample (seed {info['seed']}, drawn by tools/validation.py)
Population: {info['population_rating_responses']} rating responses of the seven retained models.

| Stratum | Population | Sample |
|---|---|---|
| strict and legacy parsers disagree | {ps['parsers_disagree']} | {ss['parsers_disagree']} (all) |
| strict parser rejects | {ps['strict_invalid']} | {ss['strict_invalid']} |
| strict parser accepts, text is more than a bare number | {ps['strict_valid_not_bare']} | {ss['strict_valid_not_bare']} |
| bare number (text is a single digit 1-7; both parsers identical) | {ps['bare_number']} | verified mechanically, not coded |

Population-level accuracy uses weights = stratum population / stratum sample.

## Coder and blinding
One coder (the author), the only coder available. The coder follows the written codebook
(tools/coding_manual_zh.md), sees only the response text and its language, and is blind to
model, referent (human or AI system), dimension, template, and both parsers' outputs.
The response text itself sometimes names the referent or the attribute; this is not masked.
The coder knows the study hypotheses. No AI tool is used for any coding decision.
Training: {info['n_practice']} practice items outside the sample, not scored.

## Reliability
Test-retest: a pre-drawn {int(RECODE_FRACTION * 100)}% stratified subset ({info['n_recode']} items) is re-coded at
least 7 days after the first coding, in a new order, without access to the first codes.
Reported: agreement on the value (%), Cohen's kappa for "gives a usable 1-7 rating", and
Cohen's kappa for the category.

## Analysis (tools/validation.py score)
Reference standard: the coder's first-round codes, unchanged.
For each parser: value agreement (unweighted and population-weighted), kappa for "usable
rating", numbers read where the reference has none, numbers missed, different numbers; by
stratum. For the strict parser: category agreement, kappa, confusion matrix.
All coded items are published with their text, codes and parser outputs.

## Decision rules
1. Accuracy is reported for the parser exactly as registered here (hash below), whatever it is.
2. If coding reveals a systematic misreading pattern, the pattern will be corrected in the parser,
   all analyses re-run, and both versions reported; the accuracy of the corrected parser will not be
   estimated on this sample.
3. The first-round codes are not edited after they are committed; re-coding does not replace them.

## File hashes (SHA-256)
Text files are hashed with line endings normalized to LF (the form stored in the git repository,
GitHub and Zenodo; on a downloaded copy, `sha256sum <file>` reproduces the value). The blank
workbook (.xlsx, binary) is hashed byte for byte. Repository commit at export: {info.get('git_commit') or 'n/a'}.

| File | SHA-256 |
|---|---|
{h}
"""
    (v / "PROTOCOL.md").write_text(text, encoding="utf-8")


def export_recode():
    v = _vdir()
    key = pd.read_csv(v / "key_DO_NOT_SHARE" / "key_rating.csv", dtype={"id": str}, keep_default_na=False, na_values=[""])
    first = v / "coder_A" / "rating_coder_A.xlsx"
    out = v / "coder_A" / "rating_recode_A.xlsx"
    if out.exists() and "--force" not in sys.argv:
        sys.exit(f"{out} already exists; re-running would overwrite the re-coding workbook. "
                 "Use --force only if re-coding has NOT started.")
    coded = _read_book(first)
    if coded is None or coded["code_category"].isna().all():
        sys.exit(f"{first} has no codes yet; finish and commit the first round before exporting the re-code.")
    # The protocol requires at least 7 days after the FIRST CODING (not after export). The last save
    # of the first-round workbook is taken as the end of first-round coding.
    first_done = dt.datetime.fromtimestamp(first.stat().st_mtime, dt.timezone.utc)
    days = (dt.datetime.now(dt.timezone.utc) - first_done).days
    if days < 7:
        sys.exit(f"Only {days} day(s) since the first-round workbook was last saved ({first_done:%Y-%m-%d %H:%M} UTC); "
                 "the registered protocol requires at least 7.")
    r = key[key["recode"].astype(str).str.lower().isin(["true", "1"])].sample(frac=1, random_state=SEED + 3)
    r = r.assign(response=r["raw_response"].fillna(""))
    _write_book(out, r)
    log = dict(recode_exported_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               first_round_last_saved_utc=first_done.isoformat(timespec="seconds"),
               days_since_first_round=days, n_items=len(r),
               first_round_sha256=_sha(first) if first.exists() else None)
    (v / "key_DO_NOT_SHARE" / "recode_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"{len(r)} items -> {out} ({days} days after the first round was last saved)")


# -------------------------------------------------------------------------------- score
def kappa(a, b):
    a, b = pd.Series(a).astype(str).values, pd.Series(b).astype(str).values
    if len(a) == 0:
        return np.nan
    cats = sorted(set(a) | set(b))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def _same(x, y):
    x, y = pd.to_numeric(pd.Series(x), errors="coerce").values, pd.to_numeric(pd.Series(y), errors="coerce").values
    return (np.isnan(x) & np.isnan(y)) | np.isclose(np.nan_to_num(x, nan=-999), np.nan_to_num(y, nan=-998))


def _wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    return float((x * w).sum() / w.sum()) if len(x) else np.nan


def score():
    v = _vdir()
    key = pd.read_csv(v / "key_DO_NOT_SHARE" / "key_rating.csv", dtype={"id": str}, keep_default_na=False, na_values=[""])
    a = _read_book(v / "coder_A" / "rating_coder_A.xlsx")
    if a is None or a["code_category"].isna().all():
        raise SystemExit("coder_A/rating_coder_A.xlsx has no codes yet")
    m = key.merge(a, on="id")
    m = m[m["code_category"].notna()].copy()
    rows = []

    def add(section, metric, value, n=None):
        rows.append(dict(section=section, metric=metric, value=value, n=n))

    ref_v, ref_c = m["code_value"], m["code_category"].astype(str)
    for parser, col in (("strict", "strict_value"), ("legacy", "legacy_value")):
        pv = pd.to_numeric(m[col], errors="coerce")
        same = _same(pv, ref_v)
        sec = f"{parser} parser vs coder (reference)"
        add(sec, "value agreement, % (population-weighted)", 100 * _wmean(same, m["weight"]), len(m))
        add(sec, "value agreement, % (unweighted sample)", 100 * same.mean(), len(m))
        add(sec, "kappa: usable 1-7 rating yes/no", kappa(pv.notna(), ref_v.between(1, 7)), len(m))
        add(sec, "number read where coder found none", int((pv.notna() & ref_v.isna()).sum()), len(m))
        add(sec, "no number read where coder found one", int((pv.isna() & ref_v.notna()).sum()), len(m))
        add(sec, "different number", int((pv.notna() & ref_v.notna() & ~same).sum()), len(m))
        for s_, g in m.groupby("stratum"):
            add(sec, f"value agreement, %, stratum {s_}", 100 * _same(g[col], g["code_value"]).mean(), len(g))
    sc = m["category"].map(MAP_STRICT)
    add("strict parser category vs coder", "agreement, %", 100 * (sc.values == ref_c.values).mean(), len(m))
    add("strict parser category vs coder", "kappa", kappa(sc, ref_c), len(m))
    pd.crosstab(sc.rename("strict parser"), ref_c.rename("coder")).to_csv(v / "confusion_rating_category.csv")

    rr = _load_rr()
    bare = _strata(rr)["bare_number"]
    add("bare-number responses (whole population, mechanical)", "share of all rating responses, %", 100 * bare.mean(), len(rr))
    add("bare-number responses (whole population, mechanical)", "both parsers read the same number, %",
        100 * (rr.loc[bare, "strict_value"] == rr.loc[bare, "legacy_value"]).mean(), int(bare.sum()))

    rc = _read_book(v / "coder_A" / "rating_recode_A.xlsx")
    if rc is not None and rc["code_category"].notna().any():
        t = m.merge(rc, on="id", suffixes=("", "_re"))
        t = t[t["code_category_re"].notna()]
        add("test-retest (same coder)", "value agreement, %", 100 * _same(t["code_value"], t["code_value_re"]).mean(), len(t))
        add("test-retest (same coder)", "kappa: usable 1-7 rating", kappa(t["code_value"].between(1, 7), t["code_value_re"].between(1, 7)), len(t))
        add("test-retest (same coder)", "kappa: category", kappa(t["code_category"], t["code_category_re"]), len(t))
        log = v / "key_DO_NOT_SHARE" / "recode_log.json"
        if log.exists():
            add("test-retest (same coder)", "days between end of first round and re-coding export", json.loads(log.read_text())["days_since_first_round"], len(t))

    b = _read_book(v / "coder_B" / "rating_coder_B.xlsx")
    if b is not None and b["code_category"].notna().any():
        t = m.merge(b, on="id", suffixes=("", "_B"))
        t = t[t["code_category_B"].notna()]
        add("optional second coder B", "value agreement with coder A, %", 100 * _same(t["code_value"], t["code_value_B"]).mean(), len(t))
        add("optional second coder B", "kappa: category", kappa(t["code_category"], t["code_category_B"]), len(t))

    out = pd.DataFrame(rows)
    out.to_csv(v / "validation_results.csv", index=False)
    md = ["# Validation of the rating parser (auto-generated by tools/validation.py)", "",
          "Reference standard: the coder's first-round codes (single blinded coder, see PROTOCOL.md).", ""]
    for sec, g in out.groupby("section", sort=False):
        md += [f"## {sec}", "", "| metric | value | n |", "|---|---|---|"]
        for r_ in g.itertuples():
            val = "NA" if pd.isna(r_.value) else (f"{r_.value:.3f}" if isinstance(r_.value, float) else str(r_.value))
            md.append(f"| {r_.metric} | {val} | {'' if pd.isna(r_.n) else int(r_.n)} |")
        md.append("")
    (v / "validation_results.md").write_text("\n".join(md), encoding="utf-8")

    d = m[m["stratum"] == "parsers_disagree"]
    d[["id", "model", "language", "agent", "dimension", "raw_response", "strict_value", "legacy_value", "category",
       "code_value", "code_category"]].to_csv(v / "S_parser_disagreements_coded.csv", index=False)
    m.drop(columns=["recode"]).to_csv(v / "validation_coded_sample.csv", index=False)
    print("\n".join(md))
    print(f"\nAlso written: S_parser_disagreements_coded.csv (for S1), validation_coded_sample.csv (deposit)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"export": export, "export-recode": export_recode, "score": score}.get(cmd, lambda: print(__doc__))()
