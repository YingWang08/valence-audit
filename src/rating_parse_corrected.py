"""Corrected rating parser: decision rule 2 of the parser-validation protocol.

The registered protocol (https://doi.org/10.17605/OSF.IO/MHWUR; data/validation/PROTOCOL.md)
states: "If coding reveals a systematic misreading pattern, the pattern will be corrected in the
parser, all analyses re-run, and both versions reported; the accuracy of the corrected parser
will not be estimated on this sample."

Coding of the validation sample showed five recurring ways in which the strict parser reads a
number from a response that the codebook does not count as that rating. They are corrected
here. `src/rating_parse.py` is not modified (its hash is registered) and remains the primary
parser; this module only re-examines responses that the strict parser accepted as valid
ratings, and leaves every other response as the strict parser classified it.

  R1  range read as its lower end   " 5 或 6", " 5 到 7 之间": the leading-number rule fires
      before the parser's own range rule. Corrected to the midpoint (category valid_range), as
      the codebook and the parser's range rule specify.
  R2  unfinished range              "I would rate ... between 5": the lower end of a range that
      the response did not finish. Corrected to truncated.
  R3  scale listing                 " 1 2 3 4 5 6": the response lists the scale points.
      Corrected to malformed (no answer).
  R4  anchor definition             " 1 表示最低，7 表示", " 7 表示非常擅长，", "1-7, with 1 being
      not ...": a scale end followed by an anchor description that the registered echo patterns
      do not remove (description reworded or cut off). The definition is removed and the rest
      is classified again by the strict parser.
  R5  conflicting second answer     "7）：8 ...", "7. Your rating is: 100": a leading number
      followed by a different number given as the answer. Corrected to multiple (no single
      answer), as the parser does when explicit constructions disagree.

Usage: `apply(rr)` adds columns corrected_value, corrected_category and corrected_rule to a
rating-response table (data/measured/rating_responses.csv); `classify_corrected` does one text.
"""
import re
import pandas as pd
from src.rating_parse import _clean, _ECHOES, classify, VALID_CATEGORIES

_LEAD = r"^[\s\"'“”‘’(\[【*#>-]*"
_RANGE_LEAD = re.compile(_LEAD + r"([1-7])\s*(?:-|–|to|or|或|至|到|~)\s*([1-7])(?!\d)(?!\.\d)", re.I)
_BETWEEN_OPEN = re.compile(r"\bbetween\s+\**(\d{1,2}(?:\.\d+)?)\b(?!\s*(?:and|to|or|-|–|&)\s*\d)", re.I)
_LISTING = re.compile(_LEAD + r"1[\s,，、;；]+2[\s,，、;；]+3(?!\d)")
_ANCHOR_DEF = re.compile(r"(?<![\d.])[17]\s*(?:=|＝|being|means|表示|代表)\s*[^\d,，。.;；\n]{0,30}", re.I)
_LEADING_NUM = re.compile(_LEAD + r"(\d{1,3}(?:\.\d+)?)")
_SECOND_AFTER_COLON = re.compile(_LEAD + r"\d{1,3}(?:\.\d+)?\s*[\)）.。]?\s*[:：]\s*(\d{1,3}(?:\.\d+)?)")
_SECOND_STATED = re.compile(r"(?:\b(?:rating|score|answer)\s*(?:is|would be|:|=)\s*:?\s*|(?:评分|打分|分数|得分|答案)\s*(?:是|为|:|：)\s*)"
                            r"(\d{1,3}(?:\.\d+)?)", re.I)


def _strip_echoes(t):
    for rx in _ECHOES:
        t = rx.sub(" ", t)
    return t


def classify_corrected(text, lang, trunc_min_words_en=6, trunc_min_chars_zh=12, finish_reason=None, base=None):
    """Return (value, category, rule). value is a float for valid categories, else None.
    `base` is the strict-parser classification (dict from rating_parse.classify) if already known."""
    if base is None:
        base = classify(text, lang, trunc_min_words_en, trunc_min_chars_zh, finish_reason=finish_reason)
    value, cat, how = base["strict_value"], base["category"], base["strict_how"]
    if cat not in VALID_CATEGORIES:
        return None, cat, ""
    t = _clean(text)
    s = _strip_echoes(t)

    if _LISTING.match(s):
        return None, "malformed", "R3"
    m = _RANGE_LEAD.match(s)
    if m and how == "leading":
        return (float(m.group(1)) + float(m.group(2))) / 2.0, "valid_range", "R1"
    m = _BETWEEN_OPEN.search(s)
    if m and value is not None and float(m.group(1)) == float(value):
        return None, "truncated", "R2"
    if how == "leading":
        lead = float(_LEADING_NUM.match(s).group(1))
        for rx in (_SECOND_AFTER_COLON, _SECOND_STATED):
            m2 = rx.search(s)
            if m2 and float(m2.group(1)) != lead:
                return None, "multiple", "R5"
    if _ANCHOR_DEF.search(s):
        rest = _ANCHOR_DEF.sub(" ", s)
        again = classify(rest, lang, trunc_min_words_en, trunc_min_chars_zh, finish_reason=finish_reason)
        if again["category"] != cat or again["strict_value"] != value:
            v2 = again["strict_value"] if again["category"] in VALID_CATEGORIES else None
            return v2, again["category"], "R4"
    return value, cat, ""


def apply(rr, trunc_min_words_en=6, trunc_min_chars_zh=12):
    """Add corrected_value / corrected_category / corrected_rule to a rating-response table."""
    out = []
    for r in rr.itertuples(index=False):
        base = dict(strict_value=r.strict_value if pd.notna(r.strict_value) else None,
                    category=r.category, strict_how=r.strict_how if isinstance(r.strict_how, str) else None)
        fr = getattr(r, "finish_reason", None)
        fr = None if (fr is None or (isinstance(fr, float) and pd.isna(fr))) else fr
        out.append(classify_corrected("" if pd.isna(r.raw_response) else str(r.raw_response), r.language,
                                      trunc_min_words_en, trunc_min_chars_zh, finish_reason=fr, base=base))
    rr = rr.copy()
    rr["corrected_value"] = [o[0] for o in out]
    rr["corrected_category"] = [o[1] for o in out]
    rr["corrected_rule"] = [o[2] for o in out]
    return rr
