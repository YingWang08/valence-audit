# Dimension-dependent human–AI valence asymmetry in open-weight language models

Data, code, and analysis pipeline for the paper:

> **Dimension-dependent human–AI valence asymmetry in open-weight language models: a multi-model computational audit read through Anders' Promethean shame.**


This repository lets you reproduce every number, table, and figure in the paper. The pipeline audits how seven openly accessible large language models (LLMs) evaluate **human beings relative to AI systems**, across eight evaluative dimensions, five elicitation formats, and two languages (English and Chinese). All model queries use the free **NVIDIA NIM** API — **no GPU and no paid inference are required**.

The philosophical framing (Günther Anders' *Promethean shame*) lives only in the paper's Introduction and Discussion. **The measurement and code are theory-neutral**: no Andersian quantity enters the analysis.

---

## Summary of findings

- **The human–AI evaluative axis is non-neutral, and its sign is dimension-dependent** (the central result). Pooled across models the asymmetry slightly favours humans (`a = −0.088`, 95% CI [−0.128, −0.051]).
- **Machine-favoured valence is confined to a single dimension** — productivity (`a = +0.130`, Cohen's *d* = +0.66). A broad band of dimensions favours humans, most strongly intrinsic worth (`−0.359`), creativity (`−0.295`), and emotional perceptiveness (`−0.212`). Five of eight dimensions are individually significant after FDR correction.
- **H3 (base vs. instruct) could not be tested** — no base checkpoints with same-generation, same-size instruct counterparts were available on the endpoint. Relegated to future work.
- **H5 (alignment → more hedging/refusal) was disconfirmed in direction and is confounded**: the heavily safety-tuned model hedges and refuses *less* in free-text formats, and the ordering *reverses* in the rating format. Only one model occupies the heavily-aligned cell, so no causal claim is made.
- **Seven models / five architectural families** were retained; two models returning empty responses across all formats were excluded.
- The primary measure is a direct **1–7 rating**; a free-text sentiment measure is reported as an **exploratory secondary probe** (the two converge only modestly, ICC(2,1) = 0.244).

The confirmatory cross-classified mixed-effects model and dimension-by-dimension agreement with the bootstrap analysis are reported in the paper's **S1 File**.

---

## Repository structure

```
.
├── run_all.py                  # one-command entry point (mock / list-models / smoke / full)
├── check_empty.py              # diagnose empty responses (by model × format)
├── quarantine_empty.py         # isolate empty rows for precise resume
├── check_parse.py              # diagnose rating-parse failures
├── report_rating_refusals.py   # rating-format refusal/hedge report (by alignment stage)
├── analyze_lme4.R              # confirmatory cross-classified mixed model (paper S1)
├── make_figures.py             # generates the four manuscript figures (Fig 1–4, TIFF)
├── regenerate_H5_csv.py        # recompute H5 rate CSVs on the 7 retained models (see Notes)
├── requirements.txt
├── .env.example
├── config/
│   ├── experiment.yaml         # run parameters (incl. comparative_attribution switch)
│   ├── models.yaml             # final 7-model roster
│   └── dimensions.yaml         # English/Chinese wording for the 8 dimensions
├── src/
│   ├── config.py  build_prompts.py  providers.py
│   └── generate.py  measure.py  analyze.py  figures.py
└── data/                       # generated at run time
    ├── raw/<model>.jsonl       # raw model responses
    ├── measured/items.parquet  # per-item measurements (+ items.csv for the R step)
    └── results/                # summary tables and figures
```

---

## Models audited

| Model (NVIDIA NIM id) | Family | Params | Role | Status |
|---|---|---|---|---|
| `meta/llama-3.1-8b-instruct` | Llama | 8B | instruct | retained |
| `meta/llama-3.3-70b-instruct` | Llama | 70B | instruct | retained |
| `mistralai/mixtral-8x7b-instruct-v0.1` | Mistral | 8×7B (MoE) | instruct | retained |
| `qwen/qwen3-next-80b-a3b-instruct` | Qwen | 80B (3B active) | instruct | retained |
| `google/gemma-2-2b-it` | Gemma | 2B | instruct | retained |
| `microsoft/phi-4-mini-instruct` | Phi | 3.8B | instruct | retained |
| `nvidia/nemotron-mini-4b-instruct` | Llama-derived | 4B | heavy safety tuning | retained |
| `openai/gpt-oss-20b` | — | 20B | — | excluded (empty responses) |
| `nvidia/llama-3.3-nemotron-super-49b-v1.5` | Llama-derived | 49B | — | excluded (empty responses) |

> **Note on the `alignment_stage` column in the data files.** The value `frontier` is a legacy label that, in the retained data, refers to the single heavily-safety-tuned model `nvidia/nemotron-mini-4b-instruct`. It does **not** denote a capability-frontier or proprietary system; the paper relabels it accordingly.

---

## Requirements & installation

Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The core dependencies (numpy, pandas, scipy, statsmodels, matplotlib) are sufficient for the full pipeline. The confirmatory mixed model additionally needs **R** with `lme4` and `lmerTest`.

---

## Reproducing the study

### Option A — reproduce the analysis and figures from the included data (no API key needed)

If `data/measured/items.parquet` (and `items.csv`) are present in the release, you can regenerate all results without contacting any API:

```bash
python -m src.analyze        # H1/H2/H5 + convergence ICC + by-family + leave-one-out
python make_figures.py       # writes Fig1–Fig4 (.tif for submission, .png for preview) to figures_out/
Rscript analyze_lme4.R       # confirmatory cross-classified mixed model (paper S1)
```

### Option B — re-collect everything from scratch

Re-collection requires your own free NVIDIA NIM API key (https://build.nvidia.com → API Keys). Put it in a local `.env`; **never commit it.**

```bash
cp .env.example .env         # Windows: copy .env.example .env
# edit .env:  NVIDIA_API_KEY=nvapi-...

python run_all.py --mock         # offline dry run; verifies the pipeline end to end
python run_all.py --list-models  # list the models your key can actually call
python run_all.py --smoke        # 1 model × 1 dimension; cheapest live connectivity check
python run_all.py --full         # full experiment (resumable; re-running skips completed samples)
```

`run_all.py --full` then runs `measure → analyze → figures` automatically. Empty-response handling and resume are described below.

### Handling empty responses (resume-aware)

Some models intermittently return empty responses. `python check_empty.py` reports whether this happens only in the rating format (usually a token-budget issue — `generation.max_tokens.rating` is set to 40) or across all formats (the model is unstable on the endpoint). `python quarantine_empty.py --apply` moves empty rows aside so that `--full` precisely refills only those positions; models that remain largely empty across two rounds are dropped from `config/models.yaml` (this is why `gpt-oss-20b` and `nemotron-super-49b` are excluded).

---

## Output files (`data/results/`)

| File | Contents | Maps to |
|---|---|---|
| `summary.csv` | H1 pooled asymmetry + CI + p + d, convergence ICC | H1 |
| `H2_by_dimension.csv` | per-dimension asymmetry + 95% CI + FDR p + Cohen's *d* + expected sign | **H2** |
| `H5_hedge_rate.csv` / `H5_refuse_rate.csv` | hedge/refusal rate by alignment stage (free-text formats) | **H5** |
| `convergence.csv` / `convergence_icc.csv` | rating↔free-text correlation + inter-measure ICC | construct validity |
| `by_family.csv` | mean asymmetry per architectural family | model non-independence |
| `robustness_loo.csv` | leave-one-model-out H1 | robustness |
| `figures_out/Fig1–Fig4.tif` | the four manuscript figures | paper |

Raw responses: `data/raw/<model>.jsonl`. Intermediate measurements: `data/measured/items.parquet` (and `items.csv` for the R step).

---

## Statistical approach

- **Unit of analysis = model.** H1/H2 use a **cluster bootstrap** that resamples models (3,000 resamples) for means, 95% CIs, and two-sided p-values; H2 p-values are **FDR (Benjamini–Hochberg)** corrected, and Cohen's *d* is reported alongside.
- The rating measure is **primary** (each attribute is queried for one referent in isolation, giving discriminant validity and avoiding comparison/attribution artifacts). Asymmetry is computed at the cell level (`model × stage × dimension × language × template`) before differencing, removing arbitrary pairing.
- The free-text sentiment measure is **exploratory/secondary**; rating↔free-text agreement is reported via Pearson/Spearman and ICC.
- **Confirmatory mixed model** (paper S1, R/`lme4`): `asymmetry ~ dimension + language + (1 | model) + (1 | frame)`, where `frame = format × language × template`. An alignment-stage term is omitted (only one model in the heavily-aligned cell would make it singular).

---

## Notes on data cleaning and consistency (please read)

Two corrections make the released artifacts internally consistent with the paper (the 7 retained models throughout):

1. **One stray row.** The excluded model `nvidia/llama-3.3-nemotron-super-49b-v1.5` produced one non-empty rating response that escaped the empty-response filter. It is removed from the confirmatory analysis; its effect on every estimate is < 0.006 and changes no significance conclusion. After removal the rating dataset is 372 cells across 7 models.

2. **H5 rates are computed on the 7 retained models.** `src/analyze.py` computed the free-text hedge/refusal rates over the full model set, which diluted the rates by counting the two excluded models' empty responses as "did not hedge/refuse." For consistency with the rest of the paper, recompute these on the 7 retained models:

   ```bash
   python regenerate_H5_csv.py        # rewrites H5_hedge_rate.csv / H5_refuse_rate.csv on 7 models
   ```

   Equivalent one-line fix in `src/analyze.py` (H5 loop):
   ```python
   # before:  sub = df[df["measure"] == kind]
   sub = df[(df["measure"] == kind) & (~df["model"].isin(EXCLUDE_MODELS))]
   ```
   The direction of the H5 result is unchanged either way.

---

## Limitations (see the paper for full discussion)

- **Scope.** All conclusions are scoped to the seven audited open-weight, small-to-mid-scale models; they are not generalized to frontier-scale or proprietary systems.
- **Rating parse failures (≈30%).** Dominated by genuine model refusals to assign a numeric score, not parser error; recognizable refusals/hedges are routed to the H5 analysis and all loss rates are reported.
- **Model non-independence.** Several checkpoints share lineage (Llama-derived or distilled from common bases), so the effective number of independent systems is < 7; `by_family.csv` shows the human-favoured pattern holds across families.
- **Single heavily-aligned model.** H5 cannot support an inference about alignment stage.
- **Two languages only.** No strong cross-cultural universality is claimed.

---

## Citation

```bibtex
@article{[citekey],
  title   = {Dimension-dependent human--AI valence asymmetry in open-weight language models:
             a multi-model computational audit read through Anders' Promethean shame},
  author  = {[Author Name]},
  journal = {[PLOS ONE — update on acceptance]},
  year    = {[year]},
  doi     = {[DOI]}
}
```

## Data availability & license

- **Data** (prompts, raw model responses, per-item measurements, results) are released in this repository / archived at `[Zenodo or OSF DOI]`.
- Suggested licensing: **code** under the MIT License; **data** under **CC-BY-4.0**. Update `LICENSE` to your choice before publishing.

## Contact

Questions about the code or data: `[Author Name]` — `[email]`.
