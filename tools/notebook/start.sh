#!/usr/bin/env bash
# One command for the GPU notebook:
#     bash tools/notebook/start.sh
# - ModelScope 192 GB AMD instance (image with ROCm, PyTorch and vLLM preinstalled): uses the image's
#   own Python/vLLM; runs all seven June checkpoints and the Nemotron-Super re-collection.
# - 24 GB NVIDIA instance: creates $WORK/venv with a pinned vLLM on first run (about 10 minutes);
#   runs the four small checkpoints (larger ones are skipped for lack of GPU memory).
# Then tools/notebook/run_models.py: download -> verify -> serve -> collect -> pack. Resumable:
# run the same command again after an interruption; finished models are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
VLLM_VERSION="${VLLM_VERSION:-0.10.1.1}"

# work dir for model weights: prefer a local disk with room for the largest model (163 GB)
free_gb() { df -P -BG "$1" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}'; }
if [ -z "${WORK:-}" ]; then
  if [ -d /mnt ] && [ "$(free_gb /mnt || echo 0)" -ge 250 ] 2>/dev/null; then WORK=/mnt/r1work; else WORK="$(dirname "$REPO")/r1work"; fi
fi
mkdir -p "$WORK"
echo "== repository: $REPO"
echo "== work dir:   $WORK  (model weights; each model is deleted after it has been run)"
df -h "$WORK" | tail -1 || true

PY=""
if python3 - <<'PY' 2>/dev/null
import torch, vllm
assert torch.cuda.is_available()
print("== using the image's own environment: vllm", vllm.__version__, "| torch", torch.__version__,
      "| hip", getattr(torch.version, "hip", None), "| gpu", torch.cuda.get_device_name(0))
PY
then
  PY="python3"
  python3 -c "import pandas, scipy, yaml, openai" 2>/dev/null || python3 -m pip install -q -i "$PIP_INDEX" pandas scipy pyyaml openai
else
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  VENV="$WORK/venv"
  if [ ! -f "$VENV/.ready_$VLLM_VERSION" ]; then
    echo "== creating Python environment with vLLM $VLLM_VERSION (first run only; about 10 minutes)"
    BASE=""
    for c in python3.12 python3.11 python3.10 python3; do
      if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)'; then BASE="$c"; break; fi
    done
    [ -n "$BASE" ] || { echo "Need Python 3.9-3.12 for vLLM $VLLM_VERSION; none found."; exit 1; }
    if [ ! -x "$VENV/bin/python" ]; then
      "$BASE" -m venv "$VENV" 2>/dev/null || { "$BASE" -m pip install -q -i "$PIP_INDEX" virtualenv && "$BASE" -m virtualenv "$VENV"; }
    fi
    "$VENV/bin/python" -m pip install -q -i "$PIP_INDEX" --upgrade pip
    "$VENV/bin/python" -m pip install -i "$PIP_INDEX" "vllm==$VLLM_VERSION" modelscope pandas scipy pyyaml python-dotenv openai
    "$VENV/bin/python" -c "import torch, vllm; assert torch.cuda.is_available(), 'torch cannot see the GPU'; print('vllm', vllm.__version__, '| torch', torch.__version__)"
    touch "$VENV/.ready_$VLLM_VERSION"
  fi
  PY="$VENV/bin/python"
  export PATH="$VENV/bin:$PATH"
fi
export WORK
"$PY" run_all.py --test
"$PY" tools/notebook/run_models.py --work "$WORK" "$@"
