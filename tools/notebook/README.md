# GPU-notebook runner (revision round)

Serves the June checkpoints from their official weights (bf16) and collects the revision-round data
for them (`python run_all.py --collect-r1 --models <id>`):

- on the ModelScope 192 GB AMD Instinct instance (ROCm; image with PyTorch and vLLM preinstalled): all
  seven June checkpoints (parts A and C) and llama-3.3-nemotron-super-49b-v1.5 (part B);
- on a 24 GB NVIDIA instance: the four small checkpoints (larger ones are skipped for lack of memory).

      # on the notebook, after uploading notebook_bundle.zip (python -m tools.make_bundle):
      unzip -o notebook_bundle.zip && cd valence-audit && bash tools/notebook/start.sh

- Weights come from ModelScope mirrors and are checked file by file against the Hugging Face release
  listed in `manifests/` (size and SHA-256 for Git-LFS files, Git blob id for the others; if a manifest
  entry disagrees, the id listed by hf-mirror.com for the same commit is consulted and the outcome is
  logged). A model whose files do not match is not run.
- Engine: vLLM; if vLLM cannot start, vLLM with another attention backend; then `hf_server.py`
  (transformers). The engine used is recorded in `data/r1/endpoints/<model>_engine.json`.
- Results: `r1_notebook_results_<UTC>.tar.gz` in the repository folder. Import it locally with
  `python -m tools.import_notebook_results <file>` (checks the commit, never shows responses).
- Resumable: run `start.sh` again; finished models are skipped. On a fresh instance, upload the last
  result archive next to `notebook_bundle.zip` first; finished work is restored.
- Logs (GPU, driver, package versions, server command and log, sampling defaults, clock check, weight
  checks) are in `data/r1/logs/notebook_<UTC>/` and travel inside the archive.
- Nothing prints response text; do not open the data files before the parser re-coding is finished.
