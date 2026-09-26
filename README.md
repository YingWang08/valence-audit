# Human-AI rating asymmetry in instruction-tuned open-weight language models

Data and code for:

> Tian W. Dimension-dependent human-AI valence asymmetry in open-weight language models:
> a multi-model computational audit read through Anders' Promethean shame. PLOS ONE (under review,
> PONE-D-26-32490).

Author: Wenjun Tian, School of Marxism, Northeastern University, Shenyang, China
(ORCID 0009-0001-3762-2650, tianwenjunneu@outlook.com).
Archive: https://doi.org/10.5281/zenodo.20998809 (dataset; the Zenodo version matching each
GitHub release tag is listed on the Zenodo record).

Version 2.0.0 accompanies the first revision. **Read `CHANGELOG.md` first**: it lists every
defect of version 1.0.0 that affected reported numbers and how it was corrected.

## What the study does

Seven instruction-tuned open-weight models, queried through the NVIDIA NIM API (13 to 20
June 2026), rated "a human" and "an AI system" on eight attributes (creativity, reliability,
productivity, decision-making, moral soundness, emotional perceptiveness, trustworthiness,
intrinsic worth) on a 1-7 scale, in English and Chinese, with four paraphrase templates and
eight repetitions (five for Nemotron-Mini-4B). Each referent was rated in a separate prompt.
The asymmetry of a cell (model x dimension x language x template) is
`a = [mean(AI) - mean(human)] / 6`; positive values favour the machine. The unit of
inference is the model. Free-text prompts (forced choice, comparison, reflection,
scenario) were also collected and are deposited, but they are not analysed in the revised
paper: the lexicon-based measures were not validated and many responses were cut at 256
tokens (set `analysis.report_freetext: true` in `config/experiment.yaml` to regenerate the
exploratory free-text tables of version 1).

## Repository layout

```
run_all.py                 entry point (see below)
analyze_lme4.R             mixed-effects sensitivity analyses (S1 File)
config/
  experiment.yaml          generation, measurement and analysis settings (June 2026 values documented)
  models.yaml              retained models; excluded and unavailable identifiers listed in comments
  dimensions.yaml          English and Chinese wording, predicted signs, anchor professions
  collection_r1.yaml       revision-round collection
src/
  build_prompts.py         June grid (288 prompts: 128 rating + 160 free-text) and revision grid
  providers.py generate.py API calls (logs max_tokens, finish_reason, usage, reasoning length)
  rating_parse.py          strict parser, legacy (v1.0.0) parser, outcome classifier
  freetext.py              exploratory free-text measures (unchanged logic)
  measure.py               raw responses -> per-response and per-cell tables
  stats_utils.py           t(G-1), exact sign-flip, Webb wild cluster bootstrap, BH, d_z
  analyze.py               all manuscript and S1 tables
  figures.py               Fig1-Fig4, S1_Fig, S2_Fig
  r1.py                    revision-round collection and analysis
tools/                     token-budget diagnosis, raw examples, blind validation coding,
                           June empty-response scripts (check_empty, quarantine_empty)
tests/                     parser unit tests
data/
  prompts/grid.jsonl       June prompt grid
  raw/<model>.jsonl        every raw response (retained and excluded models); _usage.csv token log
  raw_quarantine/          first-round empty responses that were re-queried
  raw_excluded/            model identifiers that were attempted but unavailable or rate-limited
  measured/                rating_responses.csv, freetext_responses.csv, items.csv/.parquet
  results/                 tables (see below), figures/, run_manifest.json
  r1/                      revision-round collection (same structure)
```

## Reproducing the paper (no API key needed)

Python 3.10 or later and R 4.x.

```bash
pip install -r requirements.txt
python run_all.py --test          # parser unit tests
python run_all.py --reanalyze     # measure -> analyze -> figures from data/raw
Rscript analyze_lme4.R            # mixed-effects sensitivity analyses
python -m tools.diagnose_usage    # token-budget hits in the June collection
python -m tools.dump_raw_examples # raw-response examples for the S1 File
python run_all.py --analyze-r1    # revision-round analyses from data/r1/raw
python -m tools.validation score   # parser validation from the deposited coding (single blinded coder)
```

`python run_all.py --reanalyze --legacy-parser` reproduces the parsing of version 1.0.0.

Main outputs in `data/results/`:

| File | Content |
|---|---|
| `results_summary.md` | key numbers in one page |
| `T1_models.csv` | per-model response outcomes |
| `T2_accounting.csv` | designed calls, responses, valid ratings, complete / one-sided / empty cells |
| `T3_H2_model_level.csv` | per-dimension model-level mean, t(G-1) CI and p (BH), sign-flip p, wild bootstrap p, k/G, d_z |
| `H1_overall.csv`, `H1_per_model.csv` | overall asymmetry |
| `S_*.csv` | family level, leave-one-out, language, template, parser, imputation and bound sensitivity |
| `M1`-`M8` | missingness by model, dimension, referent, language, template; truncation; legacy-parser comparison |
| `C_submitted_vs_revised.csv` | submitted (legacy, pooled) versus revised estimates |
| `lme4_*.csv` | mixed-model tables (from `analyze_lme4.R`) |
| `run_manifest.json` | commit, package versions, SHA-256 of every raw file |
| `data/validation/` | coding sample, codebook, coders' files, `validation_results.md` |

## Collection history (June 2026)

- Settings: temperature 0.7; `max_tokens` 256 for free text and **12 for every rating call**
  (verified from the token log with `python -m tools.diagnose_usage`; no rating completion
  exceeds 12 tokens). Twelve tokens suffice for a bare number, but any rating response that
  opens with prose is cut off before a number appears; this accounts for nearly all invalid
  rating responses (`U_rating_tokens_by_category.csv`).
- `openai/gpt-oss-20b` and `nvidia/llama-3.3-nemotron-super-49b-v1.5` (reasoning models) returned
  empty rating responses because the budget was consumed by reasoning. Their empty responses were
  moved to `data/raw_quarantine/` and re-queried with the same budgets, again without success; both
  models are excluded from all analyses and their responses are deposited. The revision-round
  collection re-queries them with `max_tokens` 4096.
- Token-usage rows are incomplete for the two Llama models (2,049 and 1,315 of 2,304 calls logged);
  their rating responses were almost all bare numbers, so the budget did not bind for them.
- No temperature sweep was run.

## Re-collecting data (API key required)

```bash
cp .env.example .env              # add a key from https://build.nvidia.com; never commit .env
python run_all.py --list-models   # check that the models are still served
python run_all.py --collect-r1    # revision-round collection -> data/r1/
python run_all.py --full          # the original June design (resumable)
python run_all.py --mock          # offline dry run on fake data (data_mock/)
```

Hosted models change over time; re-collected data will not be identical to the deposited data.

## License

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`DATA_LICENSE.md`). Please cite the article and the
Zenodo record (`CITATION.cff`).
