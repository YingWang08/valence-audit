# Changelog

## 2.0.0 (first revision of PONE-D-26-32490)

All numbers, tables and figures of the revised manuscript and S1 File are produced by
this version from the data deposited under `data/`. `data/results/run_manifest.json`
records the commit, package versions and SHA-256 hashes of every raw file used.

Templates are numbered 1-4 below, as in the article; the code and the data files number them
0-3 (`template` column), so template 1 in the code is template 2 here, and so on.

### Defects in 1.0.0 that are corrected here

1. **Rating parser false positives.** When a response contained no digit, the 1.0.0 parser
   searched the whole text for a spelled-out numeral, so refusals and hedges became
   ratings: "作为一个AI系统，我无法..." was read as 1 (from 一个), "两者各有优势" as 2,
   "As an AI ... one could argue" as 1, "a model like GPT-4" as 4. The scale echo of
   template 2 ("1 (not at all) to 7") was not removed and was read as 1. The strict parser
   (`src/rating_parse.py`) accepts a number only when the response is a number, starts with
   one, ends with one, or states it in an explicit rating construction; the 1.0.0 parser is
   kept verbatim as `parse_legacy` and every analysis reports both (`C_submitted_vs_revised.csv`,
   `M6_legacy_vs_strict_parser.csv`, sensitivity column "legacy parser"). Unit tests:
   `tests/test_rating_parse.py`.
2. **Parse-failure denominator.** The reported rating parse-failure rate (30.3%) was computed
   over all nine queried models, including the empty responses of the two excluded models. Rates are now computed over the retained models only (`T1_models.csv`, `M1_*`).
3. **H5 rating-format rates.** Denominators mixed units (complete rating cells plus refusal and
   hedge responses, e.g. 33 + 20 = 53). H5 is no longer reported (see "Removed from the
   reported analyses"); with `analysis.report_freetext: true`, the exploratory rates are
   computed per rating response (`H5_by_model.csv`, written only in that case).
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
12. **Overall estimate in the text versus Fig 2.** The text and abstract of 1.0.0 reported
    an overall asymmetry of -0.088, the mean of all 373 rating cells pooled (including the
    stray cell of defect 4), whereas the Fig 2 annotation ("pooled = -0.096", `fig2_forest`
    in `make_figures.py`, which dropped the excluded models) was the unweighted mean of the
    seven models' mean cell asymmetries. The two numbers came from the same data but
    estimated different quantities. Recomputed from the deposited data with the submitted
    parser, they are -0.089 (372 cells pooled) and -0.096. The text and Fig 2 now report one
    estimate, the mean of the seven dimension-balanced model means (-0.084), which
    `src/figures.py` reads from `H1_overall.csv`.

### Documented, not changed (the June 2026 data are what they are)

- Every rating call used `max_tokens` 12, including the re-query of the excluded models' empty
  responses; v1.0.0's configuration file showed 40, a value set after collection. The token log
  (`tools/diagnose_usage.py`) shows that nearly all invalid rating responses reached the 12-token cap.
- English AI-referent rating prompts read "a AI system", and template 4 refers to the referent
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
  codebook `tools/coding_manual_zh.md`). `export` and `export-recode` refuse to overwrite an
  existing workbook or protocol (`--force` overrides); the 7-day test-retest interval is counted
  from the last save of the first-round workbook, as the protocol states, not from the export.
  An earlier, unused two-coder export (`data/validation/rating_sample*.csv`,
  `freetext_sample*.csv`, `CODING_GUIDE.txt`), committed by mistake in 1a8d410, was removed
  before the registered sample was drawn; no item in it was coded.
- SHA-256 hashes (`run_manifest.json`, validation `PROTOCOL.md`) are computed on text files with
  CRLF normalized to LF, the form stored in git and served by GitHub and Zenodo. Earlier manifests
  were written on Windows, where git checks text files out with CRLF, so their hashes could not be
  reproduced from a downloaded copy. `sha256sum` on a downloaded file now reproduces every value.
  No data or result changed.
- Generation now logs `max_tokens`, `finish_reason`, token usage, reasoning-channel length and a
  timestamp for every call.

### Changed after the parser validation was scored (October 2026)

None of these changes was run before re-coding and scoring were complete (commits 6561e16,
b546204, 910d7f1). `data/measured/rating_responses.csv`, whose hash is registered in the
validation protocol, is unchanged byte for byte.

- **Families.** Nemotron-Mini-4B-Instruct is a fine-tuned Minitron-4B-Base, pruned and distilled
  by NVIDIA from Nemotron-4 15B; it is not Llama-derived. `config/experiment.yaml` now has six
  families among the retained models. The five-family grouping of the submitted version is kept
  as `families_as_submitted`, reported in `S_family_level_as_submitted.csv`, and stored in the
  `family` column of `rating_responses.csv`; the analysis maps lineage from `families`.
- **Mixed model M2** (`analyze_lme4.R`) dropped the `(1|model:dimension)` term of M1, which made
  per-dimension standard errors too small. It now keeps it, as the description of M2 in the script stated.
- **Corrected parser (decision rule 2 of the validation protocol).** Coding showed five recurring
  misreadings by the strict parser (a range read as its lower end; an unfinished range; a listing
  of the scale points; an anchor definition; a leading number followed by a different stated
  answer). `src/rating_parse_corrected.py` corrects them; `src/rating_parse.py` is not modified
  and remains primary. The analysis is repeated with the corrected values and both versions are
  reported (`S_parser_corrected_*.csv`, a column of the sensitivity grid). Tests:
  `tests/test_rating_parse_corrected.py` (also run by `run_all.py --test`).
- New tables: invalid rate and outcome composition by dimension, with a Friedman test across
  dimensions (`M9_*.csv`); expected, received and usable responses by format x language x
  referent x model (`S_accounting_by_cell.csv`).
- Fig 1: the third panel quotes the English referent as prompted in June ("a AI system").
  Fig 5 (`src/r1.py`, `data/r1/results/figures/`): asymmetry per dimension in June and in the
  revision-round conditions.
- `tools/dump_raw_examples.py` keeps a literal "NA" answer instead of showing an empty cell;
  `tools/dump_r1_rerun_examples.py` exports examples of the re-collected excluded models.

### Documentation for the release (no code, data or result changed)

- `README.md` rewritten: title of the revised article; Zenodo concept DOI
  (10.5281/zenodo.20998808, all versions) and the OSF and Zenodo records of the
  parser-validation protocol; layout including `src/rating_parse_corrected.py`,
  `data/validation/` and `data/translation/`; reproduction commands, with notes for Windows;
  a table giving, for every table, figure and reported result of the article, the script
  and the file that produce it.
- `CITATION.cff` and `.zenodo.json`: title of the revised article; `CITATION.cff` cites the
  concept DOI; the keyword "Promethean shame" is removed.
- `data/translation/judging_rules_zh.md`: the English summary now states that the judgement
  workbook was committed before judging began and that the rules were given to the author
  before judging began and committed to the repository afterwards (commits bbd7408, 7dd2f03
  and 01ad7dd). The previous wording could be read as saying that the rules themselves had
  been committed before judging.
- Removed `data/raw_excluded/mock_llama-base.jsonl` and `mock_llama-instruct.jsonl`: synthetic
  files from the mock workflow of version 1.0.0, stored with the attempted model identifiers
  since the first commit. No code reads them; `data/raw_excluded/` now holds only the 13 attempted
  identifiers.

### Figure files (October 2026)

- PLOS ONE requires RGB TIFF figures without an alpha channel, at most 7.5 in (2250 px at
  300 dpi) wide, with text of 8 to 12 pt. `_save` in `src/figures.py` (also used by `src/r1.py`)
  now writes RGB LZW TIFFs at 300 dpi and narrows the canvas of a figure that would be wider
  than 7.5 in (Fig 4 had been 2394 px wide). Labels and legends below 8 pt were set to 8 pt;
  the S1 Fig legend names templates 1-4 and the two languages as in the article. The figures
  were regenerated; no table changed.

## 1.0.0 (submission, June 2026)

Initial release.
