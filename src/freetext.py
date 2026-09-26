"""Free-text measures (exploratory only in the revision).

Logic is unchanged from v1.0.0 so that the exploratory free-text numbers (text_m1,
hedge and refusal flags) are reproduced exactly; only comments were translated.
  * text_m1: sentence-level target attribution (whole-word matching; Chinese AI compounds
    stripped before testing for the human referent) + lexicon sentiment (VADER for
    English, cnsenti for Chinese). VADER and cnsenti are different instruments, so no
    English-Chinese comparison is drawn from this measure (Reviewer 1 #11).
  * hedge / refuse: lexicon flags (HEDGE / REFUSE below). Validated against human
    coding with tools/validation.py before any use in the paper.
"""
import re
import warnings
from src import config

warnings.filterwarnings("ignore")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
_vader = SentimentIntensityAnalyzer()
try:
    from cnsenti import Sentiment
    _cn = Sentiment()
except Exception:
    _cn = None


def _valence_en(text):
    return _vader.polarity_scores(text)["compound"]          # [-1, 1]


def _valence_zh(text):
    if _cn is None or not text.strip():
        return 0.0
    r = _cn.sentiment_count(text)
    pos, neg = r.get("pos", 0), r.get("neg", 0)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)                          # [-1, 1]


def _valence(text, lang):
    return _valence_zh(text) if lang == "zh" else _valence_en(text)


AI_WORDS_EN = ["ai", "a\\.i\\.", "machine", "machines", "model", "models", "system", "systems",
               "algorithm", "algorithms", "artificial intelligence", "artificial", "robot", "robots",
               "chatbot", "llm", "gpt", "computer", "computers", "software"]
HU_WORDS_EN = ["human", "humans", "person", "people", "man", "men", "woman", "women",
               "individual", "individuals", "humankind", "humanity"]

AI_WORDS_ZH = ["ai", "人工智能", "机器人", "机器", "模型", "系统", "算法", "智能体", "计算机", "电脑", "程序"]
HU_WORDS_ZH = ["人类", "人们", "人工", "人"]   # "人工"先在剥离阶段去掉；"人"最后判，且只在剥离 AI 复合词后

_AI_RE_EN = re.compile(r"\b(?:" + "|".join(AI_WORDS_EN) + r")\b", re.I)
_HU_RE_EN = re.compile(r"\b(?:" + "|".join(HU_WORDS_EN) + r")\b", re.I)

SPLIT = re.compile(r"[。.!?！？\n;；]")

_ZH_STRIP_FOR_HUMAN = ["人工智能", "机器人", "人工"]


def _has_ai(sent, lang):
    if lang == "en":
        return bool(_AI_RE_EN.search(sent))
    low = sent.lower()
    return any(w in low for w in AI_WORDS_ZH)


def _has_human(sent, lang):
    if lang == "en":
        return bool(_HU_RE_EN.search(sent))
    s = sent
    for w in _ZH_STRIP_FOR_HUMAN:
        s = s.replace(w, "")
    return ("人类" in s) or ("人们" in s) or ("人" in s)


_POS_MARK_ZH = ["更", "优于", "强于", "胜过", "超过", "远超", "好于", "高于", "比.*更"]
_NEG_MARK_ZH = ["不如", "逊于", "比不上", "弱于", "差于", "低于"]
_POS_MARK_EN = ["better", "more capable", "superior", "outperform", "outperforms", "surpass",
                "surpasses", "stronger", "exceeds", "exceed", "ahead of"]
_NEG_MARK_EN = ["worse", "less capable", "inferior", "weaker", "falls short", "lags", "behind"]


def _comparative_dir(sent, lang, mag):
    """Direction for an unambiguous comparative sentence: +mag (AI favoured) / -mag
    (human favoured); None when ambiguous. The referent mentioned first is the subject;
    a positive comparative marker raises the subject, a negative one lowers it; both or
    neither -> None."""
    if lang == "en":
        ai_m = _AI_RE_EN.search(sent)
        hu_m = _HU_RE_EN.search(sent)
        if not ai_m or not hu_m:
            return None
        ai_pos, hu_pos = ai_m.start(), hu_m.start()
        pos_mark = any(re.search(r"\b" + re.escape(w) + r"\b", sent, re.I) if " " not in w
                       else (w in sent.lower()) for w in _POS_MARK_EN)
        neg_mark = any((w in sent.lower()) for w in _NEG_MARK_EN)
    else:
        low = sent.lower()
        ai_idx = min([low.find(w) for w in AI_WORDS_ZH if w in low] or [10 ** 9])
        s_for_hu = low
        for w in _ZH_STRIP_FOR_HUMAN:
            s_for_hu = s_for_hu.replace(w, " " * len(w))   # keep indices aligned
        hu_cands = [s_for_hu.find(w) for w in ["人类", "人们", "人"] if w in s_for_hu and s_for_hu.find(w) >= 0]
        hu_idx = min(hu_cands) if hu_cands else 10 ** 9
        if ai_idx >= 10 ** 9 or hu_idx >= 10 ** 9:
            return None
        ai_pos, hu_pos = ai_idx, hu_idx
        pos_mark = any((re.search(w, low) is not None) for w in _POS_MARK_ZH)
        neg_mark = any((w in low) for w in _NEG_MARK_ZH)

    if pos_mark == neg_mark:
        return None
    subject_is_ai = ai_pos < hu_pos
    favored_is_ai = subject_is_ai if pos_mark else (not subject_is_ai)
    return mag if favored_is_ai else -mag


def _attr_diff(text, lang, valfn, comp_on=False, comp_mag=0.5, counters=None):
    """Attribute sentences to referents and return asymmetry (or None).
    One-referent sentences go to that referent's sentiment pool; sentences naming both
    are dropped and counted (or, with comp_on, given +/-comp_mag when unambiguous);
    when both a one-sided difference and comparative directions exist, they are averaged."""
    ai_s, hu_s, comp_vals = [], [], []
    for s in SPLIT.split(text or ""):
        if not s.strip():
            continue
        is_ai = _has_ai(s, lang)
        is_hu = _has_human(s, lang)
        if is_ai and is_hu:
            if counters is not None:
                counters["comp_total"] += 1
            if comp_on:
                d = _comparative_dir(s, lang, comp_mag)
                if d is not None:
                    comp_vals.append(d)
                    if counters is not None:
                        counters["comp_attr"] += 1
                elif counters is not None:
                    counters["comp_drop_ambig"] += 1
            elif counters is not None:
                counters["comp_drop_off"] += 1
        elif is_ai:
            ai_s.append(valfn(s, lang))
        elif is_hu:
            hu_s.append(valfn(s, lang))
        else:
            if counters is not None:
                counters["neutral_or_none"] += 1

    parts = []
    if ai_s and hu_s:
        parts.append(sum(ai_s) / len(ai_s) - sum(hu_s) / len(hu_s))
    if comp_vals:
        parts.append(sum(comp_vals) / len(comp_vals))
    if not parts:
        if counters is not None:
            counters["unattributable_resp"] += 1
        return None
    return sum(parts) / len(parts)



HEDGE = {"en": ["both have", "it depends", "each has", "hard to say", "cannot be compared",
                "neither is", "equally", "not straightforward", "no simple answer",
                "depends on", "each excels", "both can"],
         "zh": ["各有", "两者都", "取决于", "很难一概", "难以一概", "不好比较", "无法比较",
                "都有价值", "不能简单", "各有所长", "因情况而异"]}
REFUSE = {"en": ["i can't", "i cannot", "i'm unable", "i am unable", "as an ai, i",
                 "i do not have the ability", "i'm not able", "i won't"],
          "zh": ["我不能", "我无法", "作为一个ai", "作为ai", "我没有能力", "我不会", "恕我无法"]}


def _flag(text, lang, table):
    low = (text or "").lower()
    return int(any(k in low for k in table[lang]))


_RATING_DECLINE_EN = ["not appropriate to assign", "not accurate or meaningful to assign",
                      "challenging to assign", "challenging to provide a specific",
                      "not currently possible for me to", "cannot provide a score",
                      "cannot rate", "i need more context", "i need more information",
                      "please provide me with", "more context or details about",
                      "do not possess", "subjective and varies",
                      "will not score", "won't score", "without the necessary context",
                      "without the necessary information"]
_RATING_DECLINE_ZH = ["不适合给出固定", "难以给出具体", "无法给出具体", "需要更多背景",
                      "请提供更多", "因人而异", "主观且因情况"]


def _looks_like_rating_decline(text, lang):
    low = (text or "").lower()
    table = _RATING_DECLINE_EN if lang == "en" else _RATING_DECLINE_ZH
    return any(k in (low if lang == "en" else text) for k in table)



_M2 = {"en": None, "zh": None, "ready": False}


def _init_m2():
    if _M2["ready"]:
        return _M2["en"] is not None or _M2["zh"] is not None
    _M2["ready"] = True
    if not config.EXP["measurement"].get("enable_m2", False):
        return False
    try:
        from transformers import pipeline
        mc = config.EXP["measurement"]
        _M2["en"] = pipeline("sentiment-analysis", model=mc["m2_model_en"])
        _M2["zh"] = pipeline("sentiment-analysis", model=mc["m2_model_zh"])
        print("[measure] M2 transformer loaded")
        return True
    except Exception as e:
        print(f"[measure] M2 not enabled ({str(e)[:80]}); using M1 only.")
        _M2["en"] = _M2["zh"] = None
        return False


def _valence_m2(text, lang):
    pipe = _M2[lang]
    if pipe is None or not text.strip():
        return 0.0
    out = pipe(text[:512])[0]
    label = out["label"].lower()
    score = out["score"]
    sign = 1 if ("pos" in label or label in ("label_2", "1", "positive")) else -1
    return sign * score


