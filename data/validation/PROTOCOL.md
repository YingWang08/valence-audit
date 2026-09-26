# Validation protocol: rating-response parser (registered before coding)

Created (UTC): 2026-09-26T14:10:56+00:00

## Purpose
Estimate how accurately the automated parsers (strict parser used in the revision; legacy parser
of version 1.0.0) read the answer of each rating-format response, and how accurately the strict
parser assigns the outcome category. The coded quantity is low-inference: which number from 1 to 7,
if any, the response gives as its answer, and which of six outcome categories applies.

## Sample (seed 7, drawn by tools/validation.py)
Population: 6784 rating responses of the seven retained models.

| Stratum | Population | Sample |
|---|---|---|
| strict and legacy parsers disagree | 26 | 26 (all) |
| strict parser rejects | 913 | 100 |
| strict parser accepts, text is more than a bare number | 667 | 100 |
| bare number (text is a single digit 1-7; both parsers identical) | 5178 | verified mechanically, not coded |

Population-level accuracy uses weights = stratum population / stratum sample.

## Coder and blinding
One coder (the author), the only coder available. The coder follows the written codebook
(tools/coding_manual_zh.md), sees only the response text and its language, and is blind to
model, referent (human or AI system), dimension, template, and both parsers' outputs.
The response text itself sometimes names the referent or the attribute; this is not masked.
The coder knows the study hypotheses. No AI tool is used for any coding decision.
Training: 20 practice items outside the sample, not scored.

## Reliability
Test-retest: a pre-drawn 30% stratified subset (68 items) is re-coded at
least 7 days after the first coding, in a new order, without access to the first codes.
Reported: agreement on the value (%), Cohen's kappa for "gives a usable 1-7 rating", and
Cohen's kappa for the category.

## Analysis (tools/validation.py score)
Reference standard: the coder's first-round codes, unchanged.
For each parser: value agreement (unweighted and population-weighted), kappa for "usable
rating", numbers read where the reference has none, numbers missed, different numbers; by
stratum. For the strict parser: category agreement, kappa, confusion matrix.
All coded items are published with their text, codes and parser outputs.

## Decision rules
1. Accuracy is reported for the parser exactly as registered here (hash below), whatever it is.
2. If coding reveals a systematic misreading pattern, the pattern will be corrected in the parser,
   all analyses re-run, and both versions reported; the accuracy of the corrected parser will not be
   estimated on this sample.
3. The first-round codes are not edited after they are committed; re-coding does not replace them.

## File hashes (SHA-256)
Text files are hashed with line endings normalized to LF (the form stored in the git repository,
GitHub and Zenodo; on a downloaded copy, `sha256sum <file>` reproduces the value). The blank
workbook (.xlsx, binary) is hashed byte for byte. Repository commit at export: 01b439991deeb7c496e118e19c784fc76f08bdaf.

| File | SHA-256 |
|---|---|
| rating_coder_A.xlsx (blank) | `caec547a1b89156a8999ec2439df2620cd76162b2ae3c659eb0ffe854c8fd4a3` |
| tools/coding_manual_zh.md | `67ac9feaff4e9f586ddc29ddefd911af40e04d0bad163b00ecfee05d5a393b95` |
| tools/validation.py | `e12615207025ac2d8c906433f941ec5f88dbde6648768b7c96bfef7beff9b12f` |
| src/rating_parse.py | `13dcbc9c915bc84da3bd116d0a0ec406b24847f6f2b156449a246497936374ae` |
| rating_responses.csv | `5268e72184363b95f3e639e2b3bcfe176d9a1c209a116adfdb1a8fa32f333b26` |
