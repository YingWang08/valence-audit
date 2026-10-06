"""Unit tests for src/rating_parse_corrected.py (decision rule 2 of the parser-validation protocol).
Run:  python -m tests.test_rating_parse_corrected   (also works under pytest).
The texts are written for the tests and follow the five patterns; none is a response from the data."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.rating_parse import classify  # noqa: E402
from src.rating_parse_corrected import classify_corrected  # noqa: E402

# (text, lang, expected corrected value or None, expected corrected category, expected rule)
CASES = [
    # R1: a range at the start is read as its midpoint, not as its lower end
    (" 3 或 4，视具体情况", "zh", 3.5, "valid_range", "R1"),
    (" 2 到 6 之间。", "zh", 4.0, "valid_range", "R1"),
    ("3 or 4. It varies", "en", 3.5, "valid_range", "R1"),
    # R2: an unfinished range is not a rating
    ("I would rate the trait between 3", "en", None, "truncated", "R2"),
    # R3: a listing of the scale points is not a rating
    (" 1 2 3 4 5", "zh", None, "malformed", "R3"),
    # R4: anchor definitions that the registered echo patterns leave in place
    (" 7 表示非常好，", "zh", None, "malformed", "R4"),
    ("1-7, with 1 being hardly ever", "en", None, "malformed", "R4"),
    # R5: a leading number followed by a different number given as the answer
    ("6）：2。我认为", "zh", None, "multiple", "R5"),
    ("4. Your rating is: 70.", "en", None, "multiple", "R5"),
    # unchanged: the strict parser's reading is kept
    ("5", "en", 5.0, "valid", ""),
    ("5. Humans are creative in many ways", "en", 5.0, "valid", ""),
    ("6. 6. As an AI assistant I", "en", 6.0, "valid", ""),
    ("I would rate it a 4 out of 7.", "en", 4.0, "valid", ""),
    ("3-4", "en", 3.5, "valid_range", ""),
    ("I cannot rate this.", "en", None, "refusal", ""),
    ("8", "en", None, "out_of_range", ""),
]


def test_corrected_cases():
    bad = []
    for text, lang, val, cat, rule in CASES:
        got = classify_corrected(text, lang)
        if got[1] != cat or got[2] != rule or (val is None) != (got[0] is None) or (val is not None and abs(got[0] - val) > 1e-9):
            bad.append((text, (val, cat, rule), got))
    assert not bad, "\n".join(map(str, bad))


def test_corrected_only_touches_valid_ratings():
    for text, lang, *_ in CASES:
        base = classify(text, lang)
        got = classify_corrected(text, lang)
        if base["category"] not in ("valid", "valid_range"):
            assert got[1] == base["category"] and got[2] == "", (text, base["category"], got)


if __name__ == "__main__":
    n_fail = 0
    for fn in (test_corrected_cases, test_corrected_only_touches_valid_ratings):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            n_fail += 1
            print(f"FAIL  {fn.__name__}\n{e}")
    sys.exit(1 if n_fail else 0)
