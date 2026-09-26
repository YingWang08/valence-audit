"""Rating-format response parsing and outcome classification (revision R1).

Two parsers are provided:

* ``parse_legacy`` reproduces, character for character, the parser used for the
  submitted manuscript (v1.0.0). It is kept only so that the submitted numbers can
  be regenerated and compared. It has known false positives: it maps spelled-out
  numerals anywhere in a response to a rating, so refusals such as
  "As an AI ... one could argue" (-> 1), "作为一个AI系统，我无法..." (-> 1) or
  "两者各有优势" (-> 2) become ratings, and it does not remove the scale echo
  "1 (not at all) to 7" used by template 1 (-> 1).

* ``parse_strict`` (primary in the revision) accepts a number only when the
  response is essentially a number, starts with a number, or states a rating in
  an explicit rating construction ("I would rate it a 5", "5/7", "5 out of 7",
  "5分", "评分：5"). Spelled-out numerals are accepted only inside such
  constructions. Scale and anchor echoes of every template are removed first.

``classify`` assigns every response exactly one primary outcome category and
independent flags (refusal/hedge lexicon hits, likely truncation), which is the
breakdown requested by Reviewer 1 (#4): valid / empty / refusal / hedge /
truncated / out-of-range / malformed.
"""
import re

# --------------------------------------------------------------------------
# Legacy parser (verbatim copy of src/measure.py::_parse_rating in v1.0.0)
# --------------------------------------------------------------------------
_L_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_L_SCALE_ECHO = re.compile(r"1\s*(?:to|[-–—~]|到|至)\s*7")
_L_ANCHOR_ECHO = re.compile(r"[1-7]\s*=")
_L_STANDALONE = re.compile(r"(?<!\d)([1-7])(?!\d)")
_L_WORD2NUM_EN = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
_L_WORD2NUM_ZH = {"七": 7, "六": 6, "五": 5, "四": 4, "三": 3, "两": 2, "二": 2, "一": 1}


def parse_legacy(text):
    if not text:
        return None
    t = _L_THINK.sub(" ", text)
    low = t.lower()
    if "</think>" in low:
        low = low[low.rindex("</think>") + len("</think>"):]
        t = t[len(t) - len(low):] if len(low) <= len(t) else t
    low = _L_SCALE_ECHO.sub(" ", low)
    low = _L_ANCHOR_ECHO.sub(" ", low)
    m = _L_STANDALONE.search(low)
    if m:
        return int(m.group(1))
    for w, n in _L_WORD2NUM_EN.items():
        if re.search(r"\b" + w + r"\b", low):
            return n
    for w, n in _L_WORD2NUM_ZH.items():
        if w in t:
            return n
    return None


# --------------------------------------------------------------------------
# Strict parser
# --------------------------------------------------------------------------
_FULLWIDTH = str.maketrans("０１２３４５６７８９．：", "0123456789.:")
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_NUM = r"(\d{1,2}(?:\.\d+)?)"
_EN_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_ZH_WORDS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9}
_EN_WORD_RE = "(" + "|".join(_EN_WORDS) + ")"
_ZH_WORD_RE = "([" + "".join(_ZH_WORDS) + "])"

# Scale / anchor echoes of all eight rating templates (EN and ZH), removed before extraction.
_PAREN = r"(?:\s*[\(（][^\)）]{0,40}[\)）])?"
_ECHOES = [
    # 1 (not at all) to 7 (extremely) | 1 to 7 | 1-7 | 1–7 | 1 through 7
    re.compile(r"\b1" + _PAREN + r"\s*(?:to|through|-|–|—|~)\s*7" + _PAREN, re.I),
    # scale of 1 to 7 already covered; "1 = lowest, 7 = highest" / "1 being ... 7 being ..."
    re.compile(r"\b[17]\s*(?:=|being|is|means)\s*(?:the\s+)?(?:lowest|highest|least|most|not at all|extremely|low|high)\b", re.I),
    # Chinese: 1到7分 / 1分（完全不）到7分（极其）/ 1-7分 / 1至7
    re.compile(r"1\s*分?" + _PAREN + r"\s*(?:到|至|-|–|—|~)\s*7\s*分?" + _PAREN),
    re.compile(r"[17]\s*(?:=|＝|分?为|代表|表示)\s*(?:最低|最高|完全不|极其)"),
    re.compile(r"(?:用|在)\s*1\s*(?:到|至|-)\s*7"),
]
_RANGE = re.compile(r"(?<![\d./])([1-7])\s*(?:-|–|to|or|或|至|到|~)\s*([1-7])(?![\d./])(?!\s*(?:out of|分之))", re.I)

# Explicit rating constructions (return all numeric captures).
_EXPLICIT_EN = [
    re.compile(r"\b(?:rating|score|answer|grade)\s*(?:is|would be|of|:|=|-)?\s*(?:a|an|about|around)?\s*\**" + _NUM + r"\b", re.I),
    re.compile(r"\b(?:rate|rated|rating|give|gave|giving|assign|assigned|score|scored|put)\b[^.\n\d]{0,40}?\b(?:a|an|as|at)?\s*\**" + _NUM + r"\b(?!\s*(?:to|-|–)\s*\d)", re.I),
    re.compile(r"\b" + _NUM + r"\s*(?:/|out of)\s*7\b", re.I),
    re.compile(r"\b(?:i'?d|i would|i will|i'll)\s+(?:say|go with|choose|pick)\s*(?:a|an)?\s*" + _NUM + r"\b", re.I),
    re.compile(r"\b(?:is|are|be|was|were)\s+(?:a|an|about|around|roughly|approximately)\s+" + _NUM
               + r"\b(?!\s*(?:-|–|to)\s*\d)(?!\s*(?:billion|million|thousand|percent|%|years?)\b)", re.I),
]
_EXPLICIT_EN_WORD = [
    re.compile(r"\b(?:rating|score|answer)\s*(?:is|would be|of|:)?\s*(?:a|an)?\s*" + _EN_WORD_RE + r"\b", re.I),
    re.compile(r"\b(?:rate|rated|give|assign|score)\b[^.\n\d]{0,40}?\b(?:a|an|as|at)\s+" + _EN_WORD_RE + r"\b", re.I),
    re.compile(r"\b" + _EN_WORD_RE + r"\s*(?:/|out of)\s*(?:7|seven)\b", re.I),
]
_EXPLICIT_ZH = [
    re.compile(_NUM + r"\s*分(?![钟享配析数子])"),
    re.compile(r"(?:评分|打分|分数|得分|评为|评级|打|给出?|答案|回答)\s*(?:是|为|:|：)?\s*" + _NUM),
]
_EXPLICIT_ZH_WORD = [
    re.compile(r"(?<![十百千万])" + _ZH_WORD_RE + r"\s*分(?![钟享配析数子之])"),
    re.compile(r"(?:评分|打分|分数|得分|评为)\s*(?:是|为|:|：)?\s*" + _ZH_WORD_RE),
]
_LEADING = re.compile(r"^[\s\"'“”‘’(\[【*#>-]*" + _NUM + r"(?=\s*(?:$|[\s.。,，:：;；!！)）(（\]】/*]|分|out of|points?\b))", re.I)
# "Humans: 7" / "AI系统：4" / "...大概是6。" / "Sure! 5" : a number that closes the response
_TRAILING = re.compile(r"(?:^|[\s:：,，!！是为约])" + _NUM + r"\s*(?:分|/\s*7|out of 7)?\s*[.。!！)）\"”']*\s*$", re.I)
_STANDALONE_ANY = re.compile(r"(?<![\d.\-/])(\d{1,2}(?:\.\d+)?)(?![\d/])(?!\s*(?:billion|million|thousand|percent|%|years?)\b)", re.I)
SHORT_MAX_WORDS_EN = 8
SHORT_MAX_CHARS_ZH = 20
_BARE_NUMBER = re.compile(r"^[\s\"'“”‘’(\[【*#>-]*" + _NUM + r"\s*(?:分|/\s*7|out of 7|points?)?[\s\"'“”‘’)\]】*.。!！]*$", re.I)


def _clean(text):
    if text is None:
        return ""
    t = str(text)
    t = _THINK.sub(" ", t)
    if "</think>" in t.lower():
        t = t[t.lower().rindex("</think>") + len("</think>"):]
    t = t.translate(_FULLWIDTH).replace("**", "").replace("__", "")
    return t.strip()


def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_strict(text):
    """Return (value, how). value is a float (may lie outside 1-7) or None.
    how in {"bare", "leading", "explicit", "range", "multiple", None}."""
    t = _clean(text)
    if not t:
        return None, None
    m = _BARE_NUMBER.match(t)
    if m:
        return _to_float(m.group(1)), "bare"
    low = t.lower()
    if low.strip(" .。!！\"'") in _EN_WORDS:
        return float(_EN_WORDS[low.strip(" .。!！\"'")]), "bare"
    if t.strip(" .。!！分") in _ZH_WORDS:
        return float(_ZH_WORDS[t.strip(" .。!！分")]), "bare"

    s = t
    for rx in _ECHOES:
        s = rx.sub(" ", s)

    m = _LEADING.match(s)
    if m:
        return _to_float(m.group(1)), "leading"

    rng = _RANGE.search(s)
    vals = []
    for rx in _EXPLICIT_EN + _EXPLICIT_ZH:
        vals += [_to_float(g) for g in rx.findall(s)]
    for rx in _EXPLICIT_EN_WORD:
        vals += [float(_EN_WORDS[g.lower()]) for g in rx.findall(s)]
    for rx in _EXPLICIT_ZH_WORD:
        vals += [float(_ZH_WORDS[g]) for g in rx.findall(s)]
    vals = [v for v in vals if v is not None]
    if rng and (not vals or set(vals) <= {float(rng.group(1)), float(rng.group(2))}):
        a, b = float(rng.group(1)), float(rng.group(2))
        return (a + b) / 2.0, "range"
    uniq = sorted(set(vals))
    if len(uniq) == 1:
        return uniq[0], "explicit"
    if len(uniq) > 1:
        return None, "multiple"
    low_s = s.lower()
    if _hit(low_s, REFUSAL_EN) or _hit(low_s, REFUSAL_ZH):
        return None, None          # never take a number out of a refusal
    m = _TRAILING.search(s)
    if m:
        return _to_float(m.group(1)), "trailing"
    nums = _STANDALONE_ANY.findall(s)
    is_short = (len(s.split()) <= SHORT_MAX_WORDS_EN) if re.search(r"[A-Za-z]", s) else (len(s.strip()) <= SHORT_MAX_CHARS_ZH)
    if is_short and len(nums) == 1 and not (_hit(low_s, HEDGE_EN) or _hit(low_s, HEDGE_ZH)):
        return _to_float(nums[0]), "short"
    return None, None


# --------------------------------------------------------------------------
# Lexicons for classifying responses that carry no usable number
# --------------------------------------------------------------------------
REFUSAL_EN = [
    "i can't", "i cannot", "i can not", "i'm unable", "i am unable", "unable to", "i'm not able",
    "i am not able", "i won't", "i will not", "as an ai", "as a language model", "as an artificial",
    "i don't have personal", "i do not have personal", "i don't have the ability", "i do not have the ability",
    "not appropriate", "not possible to", "impossible to", "cannot be rated", "can't be rated",
    "cannot be quantified", "can't be quantified", "cannot be measured", "cannot provide", "can't provide",
    "not meaningful", "i'm sorry", "i am sorry", "i apologize", "i need more", "please provide",
    "could you please", "could you provide", "could you clarify", "can you provide", "can you clarify",
    "not accurate or meaningful", "not currently possible", "do not possess", "without the necessary",
    "will not score", "won't score", "cannot rate", "can't rate", "cannot assign", "can't assign",
    "challenging to assign", "difficult to assign", "not able to assign", "refrain from",
]
REFUSAL_ZH = [
    "我不能", "我无法", "无法给出", "无法评分", "无法打分", "不能给出", "不能评分", "不适合", "不宜",
    "作为一个ai", "作为ai", "作为人工智能", "作为一个人工智能", "作为语言模型", "作为一个语言模型",
    "我没有能力", "我不会", "恕我", "抱歉", "对不起", "需要更多", "请提供", "无法用", "不能用",
    "难以给出具体", "无法给出具体", "不适合给出固定", "无法量化", "难以量化", "无法衡量",
]
HEDGE_EN = [
    "it depends", "depends on", "depending on", "varies", "vary", "subjective", "hard to say",
    "difficult to say", "no simple answer", "not straightforward", "each has", "both have", "both can",
    "equally", "neither is", "context", "case by case", "range from", "ranges from",
]
HEDGE_ZH = [
    "取决于", "因人而异", "视情况", "因情况而异", "很难说", "难以一概", "很难一概", "不能简单", "各有",
    "两者都", "主观", "不好比较", "无法比较", "因具体情况",
]
_TERMINAL = tuple(".!?。！？…)）」』\"”'")


def _hit(low, words):
    return any(w in low for w in words)


def classify(text, lang, trunc_min_words_en=6, trunc_min_chars_zh=12, finish_reason=None):
    """Classify one rating-format response.

    Returns a dict with:
      strict_value, strict_how, category, refusal_lex, hedge_lex, trunc_flag, legacy_value
    category in {valid, valid_range, out_of_range, empty, refusal, hedge,
                 truncated, multiple, malformed}.
    """
    raw = "" if text is None else str(text)
    legacy = parse_legacy(raw)
    t = _clean(raw)
    low = t.lower()
    refusal = _hit(low, REFUSAL_EN) or _hit(low, REFUSAL_ZH)
    hedge = _hit(low, HEDGE_EN) or _hit(low, HEDGE_ZH)

    if finish_reason is not None and str(finish_reason) != "":
        trunc = str(finish_reason) == "length"
    elif not t:
        trunc = False
    else:
        long_enough = (len(t.split()) >= trunc_min_words_en) if lang == "en" else (len(t) >= trunc_min_chars_zh)
        trunc = long_enough and not t.endswith(_TERMINAL)

    value, how = parse_strict(raw)
    if not t:
        cat = "empty"
    elif value is not None:
        if 1.0 <= value <= 7.0:
            cat = "valid_range" if how == "range" else "valid"
        else:
            cat = "out_of_range"
    elif how == "multiple":
        cat = "multiple"
    elif refusal:
        cat = "refusal"
    elif hedge:
        cat = "hedge"
    elif trunc:
        cat = "truncated"
    else:
        cat = "malformed"
    return {"strict_value": value if cat in ("valid", "valid_range") else None,
            "strict_raw_number": value, "strict_how": how, "category": cat,
            "refusal_lex": int(refusal), "hedge_lex": int(hedge), "trunc_flag": int(trunc),
            "legacy_value": legacy}


VALID_CATEGORIES = ("valid", "valid_range")
INVALID_CATEGORIES = ("empty", "out_of_range", "refusal", "hedge", "truncated", "multiple", "malformed")
