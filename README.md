# Human-AI rating asymmetries in instruction-tuned open-weight language models

Data and code for:

> Tian W. Human-AI rating asymmetries across evaluative dimensions in instruction-tuned
> open-weight language models: a descriptive multi-model audit. PLOS ONE (under review,
> PONE-D-26-32490).

Author: Wenjun Tian, School of Marxism, Northeastern University, Shenyang, China
(ORCID 0009-0001-3762-2650, tianwenjunneu@outlook.com).

Archive (Zenodo, all versions): https://doi.org/10.5281/zenodo.20998808. Each GitHub release
has its own Zenodo version DOI, listed on the record; release 1.0.0 is
https://doi.org/10.5281/zenodo.20998809.

Registered protocol of the parser validation: OSF, https://doi.org/10.17605/OSF.IO/MHWUR
(identical copy on Zenodo, https://doi.org/10.5281/zenodo.22992216).

Version 2.0.0 accompanies the first revision. **Read `CHANGELOG.md` first**: it lists every
defect of version 1.0.0 that affected reported numbers and how it was corrected. The
repository keeps the name of version 1.0.0 (`valence-audit`); the revised article no longer
describes the ratings as valence.

## What the study does

Seven instruction-tuned open-weight models, queried through the NVIDIA NIM API (13 to 20
June 2026), rated "a human" and "an AI system" on eight attributes (creative, reliable,
productive, good at making decisions, morally sound, emotionally perceptive, trustworthy,
intrinsically valuable) on a 1-7 scale, in English and Chinese, with four templates and
eight repetitions (five for Nemotron-Mini-4B). Each referent was rated in a separate prompt.
The asymmetry of a cell (model x dimension x language x template) is
`a = [mean(AI) - mean(human)] / 6`; positive values mean that the AI system was rated
higher, and 6a is the difference on the 1-7 scale. The unit of inference is the model, each
model weighted equally.

In September 2026 (revision round, `data/r1/`), the seven June checkpoints were served from
their official weights and answered (A) isolated rating prompts with a 150-token limit for the
June referents in the June wording, four anchor referents, the corrected English article
("an AI system") and the Chinese individual referent 一个人, and a joint condition in which
six referents were rated in one prompt on one scale (300-token limit); and (C) the June rating
prompts with the June 12-token limit. In (B), the two excluded reasoning models
answered the June rating prompts with a 4,096-token limit (gpt-oss-20b on NIM,
Nemotron-Super-49B from its official weights).

Free-text prompts (forced choice, comparison, reflection, scenario) were also collected in
June and are deposited, but they are not analysed in the revised article: the lexicon-based
measures were not validated and many responses were cut at 256 tokens (set
`analysis.report_freetext: true` in `config/experiment.yaml` to regenerate the exploratory
free-text tables of version 1).

Templates are numbered 0-3 in the code and the data files (`template` column) and 1-4 in
the article: template 1 in the code is template 2 of the article, and so on.

## Repository layout

```
run_all.py                 entry point (see below)
analyze_lme4.R             mixed models M0-M3 (Results, Table 4, S1 File)
config/
  experiment.yaml          generation, measurement and analysis settings; model families
  models.yaml              retained models; excluded and unavailable identifiers listed in comments
  dimensions.yaml          English and Chinese wording, expected directions, anchor professions
  collection_r1.yaml       revision-round collection
src/
  build_prompts.py         June grid (288 prompts: 128 rating + 160 free-text) and revision grid
  providers.py generate.py API calls (logs max_tokens, finish_reason, usage, reasoning length)
  rating_parse.py          strict parser (primary), legacy (v1.0.0) parser, outcome classifier
  rating_parse_corrected.py  corrected parser (validation decision rule 2; sensitivity analysis)
  freetext.py              exploratory free-text measures (not reported)
  measure.py               raw responses -> per-response and per-cell tables
  stats_utils.py           t(G-1), exact sign-flip, Webb wild cluster bootstrap, BH, d_z
  analyze.py               June tables (Tables 1-4, S1 File tables, run manifest)
  figures.py               Fig1, Fig2, Fig3, Fig4, S1_Fig
  r1.py                    revision-round collection and analysis (Table 5 sources, Fig5, S2_Fig)
tools/                     token-log diagnosis, raw-response examples, parser validation
                           (validation.py, coding_manual_zh.md), back-translation
                           (back_translate.py), June empty-response scripts, serving notebook
tests/                     parser unit tests (strict and corrected parser)
data/
  prompts/grid.jsonl       June prompt grid
  raw/<model>.jsonl        every June raw response (retained and excluded models); _usage.csv token log
  raw_quarantine/          first-round empty responses of the excluded models, which were re-queried
  raw_excluded/            the 13 model identifiers that were attempted but unavailable or rate-limited
  measured/                rating_responses.csv, freetext_responses.csv, items.csv/.parquet
  results/                 June tables (see below), figures/, run_manifest.json
  r1/                      revision round: prompts, raw responses, measured tables, results/ (R1_*.csv, figures/)
  validation/              parser validation: PROTOCOL.md, coded sample, coder files, results
  translation/             back-translation check: machine translations, judgement workbooks, judging rules
  model_availability/      endpoint checks of September 2026 (HTTP 410 and end-of-life dates)
  model_weights/           checks of the downloaded weight files against the Hugging Face releases
```

## Reproducing the article (no API key needed)

Python 3.10 or later and R 4.x.

```bash
pip install -r requirements.txt
python run_all.py --test          # parser unit tests
python run_all.py --reanalyze     # measure -> analyze -> figures from data/raw
Rscript analyze_lme4.R            # mixed models
python run_all.py --analyze-r1    # revision-round analyses from data/r1/raw
python -m tools.diagnose_usage    # token-limit hits in the June collection
python -m tools.dump_raw_examples # raw-response examples for the S1 File
python -m tools.validation score  # parser validation from the deposited coding (single blinded coder)
python run_all.py --mock          # offline end-to-end run on synthetic data (writes data_mock/)
```

`python run_all.py --reanalyze --legacy-parser` reproduces the parsing of version 1.0.0.

On Windows, if `Rscript` is not on the PATH, call it with its full path, for example
`& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" analyze_lme4.R` in PowerShell. Git on Windows
usually checks text files out with CRLF line endings, so after a run `git status` can list
regenerated CSV files as modified although `git diff --stat` shows no change in their content.
All SHA-256 hashes in this repository (`run_manifest.json`, `data/validation/PROTOCOL.md`) are
computed with CRLF normalized to LF; on Windows, check them with
`python -c "import hashlib;print(hashlib.sha256(open('data/measured/rating_responses.csv','rb').read().replace(b'\r\n',b'\n')).hexdigest())"`
rather than `Get-FileHash`.

## Where each result of the article comes from

Files are in `data/results/` unless another directory is given; revision-round files
(`R1_*`) are in `data/r1/results/`. Figures are written as `.tif` (submission) and `.png`.

| Article | Produced by | File(s) |
|---|---|---|
| Fig 1 | `src/figures.py` | `figures/Fig1.tif` (design diagram) |
| Table 1 | `src/analyze.py` | `T1_models.csv` (responses, invalid, categories); complete cells from `T2_accounting.csv` |
| Table 2; Results, Data accounting | `src/analyze.py` | `T2_accounting.csv`; by format, language, referent and model `S_accounting_by_cell.csv` |
| Results, Parser validation | `python -m tools.validation score` | `data/validation/validation_results.md`, `validation_results.csv`, `confusion_rating_category.csv`, `S_parser_disagreements_coded.csv` |
| Corrected parser (Parser validation; Table 4 note) | `src/analyze.py` with `src/rating_parse_corrected.py` | `S_parser_corrected_changes.csv`, `S_parser_corrected_outcomes.csv`, `S_parser_corrected_patterns.csv`, `S_parser_corrected_H2.csv` |
| H1; Fig 2 | `src/analyze.py`; `src/figures.py` | `H1_overall.csv`, `H1_per_model.csv`, `cells_primary.csv`; `figures/Fig2.tif` |
| Table 3; H2 | `src/analyze.py` | `T3_H2_model_level.csv` |
| Fig 3 | `src/figures.py` | `T3_H2_model_level.csv`, `S_model_by_dimension.csv`, `S_language_by_dimension.csv`; `figures/Fig3.tif` |
| Results, Invalid responses; Fig 4 | `src/analyze.py`; `tools/diagnose_usage.py`; `src/figures.py` | categories `M1_outcomes_by_model.csv`; 12-token limit `U_rating_tokens_by_category.csv`, `U_rating_tokens_by_model.csv`; templates `M8_invalid_by_template_referent.csv`; dimensions `M9_invalid_by_dimension.csv`, `M9b_invalid_dimension_test.csv`, `M9c_invalid_rate_model_by_dimension.csv`; referent `M3_referent_gap_by_dimension.csv`, `M2_invalid_by_model_referent_language.csv`; full cross-tabulation `M4_crosstab_model_dimension_referent_language_template.csv`; `figures/Fig4.tif` (from `data/measured/rating_responses.csv`) |
| Table 4; Results, Sensitivity analyses; S1 Fig | `src/analyze.py`; `analyze_lme4.R`; `src/figures.py` | language, template, parser, invalid-response treatments, without Mixtral and Nemotron-Mini: `S_sensitivity_wide.csv`, `S_sensitivity_long.csv`; family level `S_family_level.csv` (five families as submitted: `S_family_level_as_submitted.csv`); leave-one-out `S_loo_model.csv`, `S_loo_family.csv`; language with BH `S_language_by_dimension.csv`; M1 `lme4_rating_M1_dimension_means_fixed.csv`; `figures/S1_Fig.tif` |
| Mixed models M0-M3 | `analyze_lme4.R` | `lme4_rating_M*_fixed.csv`, `lme4_rating_M*_varcomp.csv`, `lme4_rating_M0_anova_type2.csv`, `lme4_session_info.txt` (`lme4_rating_legacy_*`: legacy parser) |
| Results, Consistency | `src/analyze.py` | `S_consistency_repeats.csv`, `S_consistency_split_half.csv`, `S_consistency_levels.csv` |
| Table 5, June 2026, 12 tokens | `src/analyze.py` | `T3_H2_model_level.csv` (`mean`, `k_same_sign`, `p_t_BH`) |
| Table 5, Revision round, 150 tokens | `src/r1.py` | `R1_replication.csv` (`mean`, `k_same_sign`, `p_t_BH`) |
| Table 5, Revision round, 12 tokens | `src/r1.py` | `R1_budget_asymmetry.csv` (`r1_12_tokens_mean`, `r1_12_tokens_k`; the asterisks are the Benjamini-Hochberg adjustment of `r1_12_tokens_p_t` over the eight dimensions) |
| Table 5, Joint rating | `src/r1.py` | `R1_joint_vs_isolated.csv` (`joint_mean`, `joint_k`, `joint_p_t_BH`) |
| Table 5, Chinese, 人类 and 一个人 | `src/r1.py` | `R1_zh_asymmetry_with_humankind.csv`, `R1_zh_asymmetry_with_individual.csv` (`mean`, `k_same_sign`, `p_t_BH`) |
| Fig 5 | `src/r1.py` | `R1_fig5_conditions.csv`; `data/r1/results/figures/Fig5.tif` |
| Results, Revision-round checks | `src/r1.py` | replication `R1_replication*.csv`, `R1_june_vs_r1_dimension_pattern.csv`; output limit `R1_budget_*.csv`; time and serving environment `R1_june_vs_r1_prompt_level*.csv`, `R1_environment.csv`; anchors `R1_referent_profiles.csv`, `R1_anchored_*.csv`, `R1_anchor_spread*.csv`; joint rating `R1_joint_*.csv`; Chinese referent `R1_zh_*.csv`; article `R1_article_effect_en.csv`; excluded reasoning models `R1_rerun_excluded_*.csv`, `R1_H2_with_rerun_models.csv`; outcomes `R1_outcomes.csv`; overview `R1_summary.md` |
| S2 Fig | `src/r1.py` | `R1_referent_profiles.csv`; `data/r1/results/figures/S2_Fig.tif` |
| S1 File, continuity with the submitted analysis | `src/analyze.py` | `C_submitted_vs_revised.csv`, `M6_legacy_vs_strict_parser.csv`, `M6b_legacy_false_positive_values.csv` |
| S1 File, raw-response examples | `tools/dump_raw_examples.py`, `tools/dump_r1_rerun_examples.py` | `S_raw_examples*`, `data/r1/results/S_r1_rerun_examples.*` |
| S1 File, back-translation check | `tools/back_translate.py`; author's judgements | `data/translation/` |
| Key numbers on one page | `src/analyze.py`; `src/r1.py` | `results_summary.md`; `R1_summary.md` |
| Provenance | `src/analyze.py` | `run_manifest.json`: commit, package versions, SHA-256 of every raw file |

## Collection history (June 2026)

- Settings: temperature 0.7; `max_tokens` 256 for free text and **12 for every rating call**
  (verified from the token log with `python -m tools.diagnose_usage`; no rating completion
  exceeds 12 tokens). Twelve tokens suffice for a bare number, but any rating response that
  opens with prose is cut off before a number appears; this accounts for nearly all invalid
  rating responses (`U_rating_tokens_by_category.csv`).
- `openai/gpt-oss-20b` and `nvidia/llama-3.3-nemotron-super-49b-v1.5` (reasoning models) returned
  empty rating responses because the budget was consumed by reasoning. Their empty responses were
  moved to `data/raw_quarantine/` and re-queried with the same budgets, again without success; both
  models are excluded from all analyses of the June data and their responses are deposited. The
  revision-round collection re-queried them with `max_tokens` 4096.
- Token-usage rows are incomplete for the two Llama models (2,049 and 1,315 of 2,304 calls logged)
  and for Mixtral-8x7B (2,300); the two Llama models' rating responses were almost all bare numbers,
  so the limit did not bind for them.
- No temperature sweep was run.

## Re-collecting data (API key required)

```bash
cp .env.example .env              # add a key from https://build.nvidia.com; never commit .env
python run_all.py --list-models   # check that the models are still served
python run_all.py --collect-r1    # revision-round collection -> data/r1/
python run_all.py --full          # the original June design (resumable)
```

Hosted models change over time; re-collected data will not be identical to the deposited data.
The seven June checkpoints were retired from the NIM endpoint between July and August 2026
(`data/model_availability/`).

## License

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`DATA_LICENSE.md`). Please cite the article and the
Zenodo record (`CITATION.cff`).
