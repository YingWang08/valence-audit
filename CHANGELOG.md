# Changelog

## 2.0.0 (first revision of PONE-D-26-32490)

All numbers, tables and figures of the revised manuscript and S1 File are produced by
this version from the data deposited under `data/`. `data/results/run_manifest.json`
records the commit, package versions and SHA-256 hashes of every raw file used.

### Defects in 1.0.0 that are corrected here

1. **Rating parser false positives.** When a response contained no digit, the 1.0.0 parser
   searched the whole text for a spelled-out numeral, so refusals and hedges became
   ratings: "作为一个AI系统，我无法..." was read as 1 (from 一个), "两者各有优势" as 2,
   "As an AI ... one could argue" as 1, "a model like GPT-4" as 4. The scale echo of
   template 1 ("1 (not at all) to 7") was not removed and was read as 1. The strict parser
   (`src/rating_parse.py`) accepts a number only when the response is a number, starts with
   one, ends with one, or states it in an explicit rating construction; the 1.0.0 parser is
   kept verbatim as `parse_legacy` and every analysis reports both (`C_submitted_vs_revised.csv`,
   `M6_legacy_vs_strict_parser.csv`, sensitivity column "legacy parser"). Unit tests:
   `tests/test_rating_parse.py`.
2. **Parse-failure denominator.** The reported rating parse-failure rate (30.3%) was computed
   over all nine queried models, including the empty responses of the two excluded models. Rates are now computed over the retained models only (`T1_models.csv`, `M1_*`).
3. **H5 rating-format rates.** Denominators mixed units (complete rating cells plus refusal and
   hedge responses, e.g. 33 + 20 = 53). Rates are now per rating response (`H5_by_model.csv`).
4. **Excluded-model rows.** One rating cell of an excluded model entered the rating data
   (373 instead of 372 cells), and the free-text hedge and refusal rates in `src/analyze.py`
   included the excluded models. Excluded models now never enter the measured tables.
5. **Weighting.** Estimates pooled cells, so models with more valid responses weighed more.
   The unit of analysis is now the model, each weighted equally.
6. **Bootstrap p-values at the resolution floor.** The pairs cluster bootstrap (3,000 draws)
   is replaced by model-level t(G-1) inference with sensitivity analyses suited to few
   clusters (`src/stats_utils.py`).
7. **Leave-one-out.** 1.0.0 computed leave-one-model-out only for the overall mean; it is now
   computed for every dimension, and leave-one-family-out is added.
8. **Temperature sweep.** `config/experiment.yaml` listed a temperature sweep that no code
   implemented and that was never run; the key is removed.
9. **Two different R scripts** (`analyze_lme4.R`, `src/analyze_lme4.R`) are merged into one;
   each reported table now comes from a single fit.
10. **Mock workflow** stopped at the measurement step because mock files were skipped; mock
    runs now write to `data_mock/` and complete end to end.
11. **Deposit.** `.gitignore` excluded `data/raw`, `data/measured`, `data/results` and
    `data/prompts`; all data are now included. Added LICENSE (MIT), DATA_LICENSE.md
    (CC BY 4.0), CITATION.cff, `.zenodo.json` (dataset, author with ORCID) and
    `.env.example`; removed IDE files.

### Documented, not changed (the June 2026 data are what they are)

- Every rating call used `max_tokens` 12, including the re-query of the excluded models' empty
  responses; v1.0.0's configuration file showed 40, a value set after collection. The token log
  (`tools/diagnose_usage.py`) shows that nearly all invalid rating responses reached the 12-token cap.
- English AI-referent rating prompts read "a AI system", and template 3 refers to the referent
  as "it". The Chinese referent 人类 denotes humans collectively, whereas "a human" denotes an
  individual. The revision-round collection tests the first and third points directly.
- Model identifiers that were attempted but unavailable or rate-limited are listed in
  `config/models.yaml`; their (mostly empty) files are in `data/raw_excluded/`.

### Removed from the reported analyses

- Free-text lexicon measures (hedging and refusal flags, VADER / cnsenti sentiment), the
  rating-versus-free-text convergence check, and the pre-specified H5 (hedging and refusal by
  alignment stage). The lexicons were never validated against human coding; many free-text
  responses were cut at 256 tokens; and rating-format "refusals" were largely produced by the
  12-token budget. The code remains (`analysis.report_freetext`), and all free-text responses are
  deposited.

### Added

- `src/measure.py`: per-response rating table with outcome categories (valid, empty, refusal,
  hedge, truncated, out-of-range, multiple, malformed) and flags; per-response free-text table.
- `src/analyze.py`: model-level H1 and H2; sign-flip and Webb wild cluster bootstrap; family
  level; leave-one-out; language, template, parser, imputation and worst-case-bound
  sensitivity; missingness decomposition by model, dimension, referent, language and
  template; design and observation accounting; exploratory H5; run manifest.
- `src/figures.py`: figures without embedded titles, model-level confidence intervals,
  missingness figure.
- `src/r1.py` and `config/collection_r1.yaml`: revision-round collection (anchor referents,
  joint rating condition, exact replication of the June prompts, re-collection of the two
  excluded reasoning models with an adequate token budget) and its analysis.
- `tools/`: token-budget diagnosis, raw-response examples, blinded coding validation of the rating
  parser: registered protocol, blinded single coder, 7-day test-retest (`tools/validation.py`,
  codebook `tools/coding_manual_zh.md`).
- Generation now logs `max_tokens`, `finish_reason`, token usage, reasoning-channel length and a
  timestamp for every call.

## 1.0.0 (submission, June 2026)

Initial release.
