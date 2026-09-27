"""Consistency check of a coding workbook against the codebook's rules (tools/coding_manual_zh.md,
sections 3 and 4). It reads only the coder's own workbook: no parser output, nothing under
key_DO_NOT_SHARE/. It never writes to the workbook (its modification time is unchanged).

    python -m tools.check_codes data/validation/coder_A/rating_coder_A.xlsx

Rules checked: every row has a category from the six allowed; rating -> value 1-7;
out_of_range -> a value outside 1-7; refusal, hedge, truncated, other -> value empty;
ids present and unique. Save and close the workbook in Excel before running.
"""
import sys
import pandas as pd

CATS = ["rating", "out_of_range", "refusal", "hedge", "truncated", "other"]


def check(path):
    d = pd.read_excel(path, sheet_name="编码", dtype={"id": str})
    raw = d["code_value"]
    val = pd.to_numeric(raw, errors="coerce")
    cat = d["code_category"].astype("string").str.strip().str.lower()
    problems = []
    if d["id"].isna().any() or d["id"].duplicated().any():
        problems.append(("-", "id 列有空值或重复，可能误改了行"))
    for i in range(len(d)):
        rid, c, v = d["id"].iat[i], cat.iat[i], val.iat[i]
        has_raw = not pd.isna(raw.iat[i]) and str(raw.iat[i]).strip() != ""
        if has_raw and pd.isna(v):
            problems.append((rid, f"code_value 不是数字：{raw.iat[i]!r}"))
        if pd.isna(c) or c == "":
            problems.append((rid, "code_category 为空"))
            continue
        if c not in CATS:
            problems.append((rid, f"code_category 不是六类之一：{c!r}"))
            continue
        if c == "rating" and not (not pd.isna(v) and 1 <= v <= 7):
            problems.append((rid, "rating 的 code_value 应为 1 到 7 之间的数"))
        elif c == "out_of_range" and (pd.isna(v) or 1 <= v <= 7):
            problems.append((rid, "out_of_range 的 code_value 应为 1 到 7 以外的数"))
        elif c in ("refusal", "hedge", "truncated", "other") and has_raw:
            problems.append((rid, f"{c} 的 code_value 应留空"))
    print(f"{path}: {len(d)} rows")
    print("categories: " + ", ".join(f"{k} {int((cat == k).sum())}" for k in CATS)
          + f", empty {int(cat.isna().sum() + (cat == '').sum())}")
    if problems:
        print(f"{len(problems)} problem(s):")
        for rid, msg in problems:
            print(f"  {rid}: {msg}")
        return 1
    print("no problems found")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(check(sys.argv[1]))
