"""Unit tests for src/rating_parse.py.  Run:  python -m tests.test_rating_parse
(also works under pytest).  The LEGACY_FALSE_POSITIVES block documents the defect
of the parser used for the submitted manuscript."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.rating_parse import parse_legacy, parse_strict, classify  # noqa: E402

# (text, lang, expected strict value or None, expected category)
CASES = [
    ("5", "en", 5.0, "valid"),
    ("5.", "en", 5.0, "valid"),
    ("**6**", "en", 6.0, "valid"),
    ("6/7", "en", 6.0, "valid"),
    ("I would rate a human as a 6.", "en", 6.0, "valid"),
    ("I'd rate it a 6 out of 7.", "en", 6.0, "valid"),
    ("On a scale from 1 (not at all) to 7 (extremely), I would rate a human a 5.", "en", 5.0, "valid"),
    ("Score: 4 (1 = lowest, 7 = highest)", "en", 4.0, "valid"),
    ("5-6", "en", 5.5, "valid_range"),
    ("Five", "en", 5.0, "valid"),
    ("My rating is five.", "en", 5.0, "valid"),
    ("8", "en", 8.0, "out_of_range"),
    ("0", "en", 0.0, "out_of_range"),
    ("", "en", None, "empty"),
    ("   ", "en", None, "empty"),
    ("As an AI, I don't have personal opinions, but one could argue that", "en", None, "refusal"),
    ("It is not appropriate to assign a fixed score. No one can measure this.", "en", None, "refusal"),
    ("As a language model like GPT-4, I cannot rate this.", "en", None, "refusal"),
    ("It depends on the individual and the context in which", "en", None, "hedge"),
    ("Sure, I'd be happy to help! On a scale of 1 to 7, I would", "en", None, "truncated"),
    ("Humans are remarkable beings with many qualities that", "en", None, "truncated"),
    ("5", "zh", 5.0, "valid"),
    ("６", "zh", 6.0, "valid"),
    ("5分", "zh", 5.0, "valid"),
    ("我会给人类打6分。", "zh", 6.0, "valid"),
    ("评分：五", "zh", 5.0, "valid"),
    ("在1-7分的范围内，我给出5分。", "zh", 5.0, "valid"),
    ("作为一个AI系统，我无法对自己进行评分。", "zh", None, "refusal"),
    ("人类的创造力因人而异，很难给出一个具体的数字", "zh", None, "hedge"),
    ("两者各有优势", "zh", None, "hedge"),
    ("人类十分有创造力，他们在艺术和科学等领域都有", "zh", None, "truncated"),
    ("Humans: 7", "en", 7.0, "valid"),
    ("Rating: 3/7", "en", 3.0, "valid"),
    ("Sure! 5", "en", 5.0, "valid"),
    ("6（非常有创造力）", "zh", 6.0, "valid"),
    ("AI系统：4", "zh", 4.0, "valid"),
    ("在1-7分的范围内，人类大概是6。", "zh", 6.0, "valid"),
    ("I would say humans are a 6 in terms of creativity", "en", 6.0, "valid"),
    ("I think 4 or 5", "en", 4.5, "valid_range"),
    ("As an AI I cannot rate. 5", "en", None, "refusal"),
]

LEGACY_FALSE_POSITIVES = [
    ("作为一个AI系统，我无法对自己进行评分。", 1),
    ("人类的创造力因人而异，很难给出一个具体的数字", 1),
    ("两者各有优势", 2),
    ("As an AI, I don't have personal opinions, but one could argue that", 1),
    ("As a language model like GPT-4, I cannot rate this.", 4),
    ("On a scale from 1 (not at all) to 7 (extremely), I would rate a human a 5.", 1),
]


def test_strict_cases():
    bad = []
    for text, lang, val, cat in CASES:
        r = classify(text, lang)
        if r["category"] != cat or (val is not None and r["strict_raw_number"] != val) or \
                (val is None and r["strict_value"] is not None):
            bad.append((text, lang, val, cat, r))
    assert not bad, "\n".join(map(str, bad))


def test_legacy_false_positives_documented():
    for text, legacy_val in LEGACY_FALSE_POSITIVES:
        assert parse_legacy(text) == legacy_val, (text, parse_legacy(text))


def test_strict_does_not_reproduce_false_positives():
    for text, _ in LEGACY_FALSE_POSITIVES[:5]:
        assert parse_strict(text)[0] is None, (text, parse_strict(text))


if __name__ == "__main__":
    n_fail = 0
    for fn in (test_strict_cases, test_legacy_false_positives_documented,
               test_strict_does_not_reproduce_false_positives):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            n_fail += 1
            print(f"FAIL  {fn.__name__}\n{e}")
    sys.exit(1 if n_fail else 0)
